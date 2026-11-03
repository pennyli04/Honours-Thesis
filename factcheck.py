"""Misinformation checker: text claims (stage 1), images (stage 2) and C2PA
provenance (stage 3).

    python factcheck.py --text "The Australian census was held on 11 August 2026"
    python factcheck.py --image photo.jpg --image-url https://example.com/photo.jpg
    python factcheck.py --image photo.jpg --caption "Flooding in Lismore, 2026" --json
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import logging
import mimetypes
import os
import re
import sys
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("factcheck")

# Configuration

MODEL = os.environ.get("FACTCHECK_MODEL", "claude-sonnet-5")
SERPAPI_KEY = os.environ.get("SERPAPI_API_KEY")
MANIPULATION_MODEL_ID = os.environ.get(
    "MANIPULATION_MODEL_ID", "prithivMLmods/Deep-Fake-Detector-v2-Model"
)

# API limit for base64 images is about 5 MB
MAX_IMAGE_BYTES = 5 * 1024 * 1024

VERDICTS = {"likely_true", "likely_false", "misleading", "unverifiable"}

# Saved with every result. Bump when a system prompt changes.
PROMPT_VERSION = "v3-central-assertion"
IMAGE_PROMPT_VERSION = "img-v4"  # stage 2 prompt; bump when it changes
PROVENANCE_PROMPT_VERSION = "prov-v2"  # stage 3 prompt; bump when it changes
CONFIDENCES = {"high", "medium", "low"}
CATEGORIES = {"content_manipulation", "miscontextualised", "authentic_truthful", "unverifiable"}

_client = None


def get_client():
    """Create the Anthropic client on first use so the module imports without a key."""
    global _client
    if _client is None:
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your environment or a .env file."
            )
        # extra retries for rate limits
        _client = Anthropic(api_key=api_key, max_retries=6)
    return _client


# Result types


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str  # likely_true | likely_false | unverifiable
    confidence: str  # high | medium | low
    justification: str
    sources: List[str] = field(default_factory=list)


@dataclass
class ImageVerdict:
    image_ref: str
    category: str  # content_manipulation | miscontextualised | authentic_truthful | unverifiable
    verdict: str
    confidence: str
    justification: str
    manipulation_label: str = ""
    manipulation_score: float = 0.0
    sources: List[str] = field(default_factory=list)


@dataclass
class ProvenanceVerdict:
    file_ref: str
    category: str
    verdict: str
    confidence: str
    justification: str
    manifest_present: bool = False
    validation_state: str = ""
    issuer: str = ""
    sources: List[str] = field(default_factory=list)


@dataclass
class Report:
    """Everything check_item() found for one piece of content."""

    claims: List[ClaimVerdict] = field(default_factory=list)
    image: Optional[ImageVerdict] = None
    provenance: Optional[ProvenanceVerdict] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# Shared helpers: text extraction, verdict parsing, the tool loop

_VERDICT_RE = re.compile(r"<verdict>\s*(.*?)\s*</verdict>", re.DOTALL)


def _text_of(content) -> str:
    """Join the text blocks of a response."""
    return "".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")


def _parse_verdict(text: str) -> Optional[dict]:
    """Pull the last <verdict>{...}</verdict> block out of the model's text."""
    matches = _VERDICT_RE.findall(text or "")
    if not matches:
        return None
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", matches[-1].strip())
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(body[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _pick(value: Any, allowed: set, default: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in allowed else default


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def new_trace() -> dict:
    """Empty trace for _run_agent to fill in (cost, retrieved URLs, raw output)."""
    return {
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "web_search_requests": 0,
        "retrieved_urls": [],
        "stop_reasons": [],
        "raw_text": "",
        "followup_used": False,
    }


def _record(trace: Optional[dict], response) -> None:
    if trace is None:
        return
    trace["api_calls"] += 1
    trace["stop_reasons"].append(getattr(response, "stop_reason", None))
    usage = getattr(response, "usage", None)
    if usage is not None:
        trace["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        trace["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        stu = getattr(usage, "server_tool_use", None)
        if stu is not None:
            trace["web_search_requests"] += getattr(stu, "web_search_requests", 0) or 0
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        results = getattr(block, "content", None)
        if isinstance(results, list):  # an error result is a single object, not a list
            for r in results:
                url = getattr(r, "url", None)
                if url and url not in trace["retrieved_urls"]:
                    trace["retrieved_urls"].append(url)


def _create(client, trace, **kwargs):
    # the API rejects an empty tools list
    if not kwargs.get("tools"):
        kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)
    response = client.messages.create(**kwargs)
    _record(trace, response)
    return response


def _run_agent(
    *,
    system: str,
    messages: List[dict],
    tools: List[dict],
    dispatch: Dict[str, Callable[[dict], dict]],
    max_turns: int,
    max_tokens: int,
    trace: Optional[dict] = None,
    first_response=None,
) -> Optional[dict]:
    """ReAct-style tool loop shared by all three checks."""
    client = get_client()
    final_text = ""

    for turn in range(max_turns):
        if turn == 0 and first_response is not None:
            response = first_response
            _record(trace, response)
        else:
            response = _create(
                client, trace,
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )
        messages.append({"role": "assistant", "content": response.content})
        final_text = _text_of(response.content)

        if response.stop_reason == "pause_turn":
            continue  # server tool still working; resend so it can carry on
        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "user", "content": _run_tools(response.content, dispatch)})
        final_text = ""  # a reply that asked for tools is not the final answer

    if trace is not None:
        trace["raw_text"] = final_text
    verdict = _parse_verdict(final_text)
    if verdict is not None:
        return verdict

    # ask once more for just the verdict
    nudge = (
        "Reply with ONLY the <verdict>...</verdict> JSON block, based on what "
        "you have already found. No other text before or after it."
    )
    last = messages[-1]
    if last["role"] == "user":
        # turn cap hit right after tool results: add the request to them
        last["content"] = list(last["content"]) + [{"type": "text", "text": nudge}]
    else:
        messages.append({"role": "user", "content": nudge})

    if trace is not None:
        trace["followup_used"] = True
    try:
        followup = _create(
            client, trace,
            model=MODEL,
            max_tokens=800,
            system=system,
            tools=tools,
            tool_choice={"type": "none"},
            messages=messages,
        )
    except Exception as exc:  # the main result is still worth returning as unverifiable
        log.warning("verdict follow-up failed: %s", exc)
        return None
    followup_text = _text_of(followup.content)
    if trace is not None:
        trace["raw_text"] = (final_text + "\n\n[follow-up]\n" + followup_text).strip()
    return _parse_verdict(followup_text)


def _run_tools(content, dispatch) -> List[dict]:
    """Run every client-side tool call in a response."""
    results = []
    for block in content:
        if getattr(block, "type", None) != "tool_use":
            continue
        handler = dispatch.get(block.name)
        if handler is None:
            result = {"error": f"unknown tool {block.name}"}
        else:
            try:
                result = handler(block.input or {})
            except Exception as exc:  # a failing tool must not kill the whole check
                log.exception("tool %s failed", block.name)
                result = {"error": f"{block.name} failed: {exc}"}
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            }
        )
    return results


# Check 1: text claims

EXTRACTION_SYSTEM_PROMPT = """You extract checkable factual claims from news text.

Rules:
- Only extract claims that could in principle be verified against an outside
  source (an event, a statistic, a quote, an action attributed to a named
  person or organisation, a date, a location).
- Do not extract opinions, predictions, or vague statements.
- Keep each claim short and self-contained: someone should be able to search
  for it without reading the original article.
- Return between 1 and 6 claims. If the text contains no checkable claims,
  return an empty list.

Respond with ONLY a JSON array of strings. No other text."""

VERIFICATION_SYSTEM_PROMPT = """You are a careful fact-checker. You will be
given a single factual claim. Use web search to find independent reporting
or primary sources that confirm, contradict, or fail to resolve the claim.

Deciding what to check:
- If the claim reports what someone said or believes ("X claimed that Y",
  "X says Y", "according to X, Y"), check whether Y is true. Confirming
  that X really said it does not make the claim true: a false statement
  accurately quoted is still a false claim. Only judge the attribution
  itself when the claim is solely about whether the statement was made
  (for example "X never said Y").
- Judge the claim as a reader would understand it, including what it
  implies, not just its most literal reading.

Process:
- Search for the claim using specific, targeted queries (names, dates,
  events) rather than pasting the whole claim as one search.
- Prefer primary sources (official statements, transcripts, original
  reporting) over aggregators or social media discussion of the claim.
- If your first search is inconclusive, try a different angle before giving
  up. Do not stop after a single search unless it clearly settles the claim.
- Note if sources disagree with each other.

Choosing a verdict:
First identify the claim's central assertion: the one thing that would have
to be true for the claim to be true. Judge that, then pick exactly one label.
- "likely_true": the evidence supports the central assertion. Extra
  context, minor imprecision or small rounding do not change this; almost
  any true claim could be given more context.
- "likely_false": the evidence contradicts the central assertion, even if
  some surrounding details are accurate. A real quote of a false statement,
  a real event with the wrong figure attached, or a real person credited
  with something they did not bring about are all "likely_false".
- "misleading": the central assertion is accurate, but the claim creates a
  false overall impression through selective framing or omitted context,
  or reliable sources genuinely conflict about the central assertion. Do
  not use it as a middle ground for a claim that is mostly false, or as a
  hedge when you are unsure.
- "unverifiable": the evidence cannot establish or refute the central
  assertion. This includes figures that official sources or fact-checkers
  describe as unproven or unsubstantiated, and figures that cannot be
  checked because the timeframe or baseline is not specified. Use it
  honestly; guessing to avoid this label defeats the point of the system.

When you have gathered enough evidence, end your response with a JSON block
in exactly this format, and nothing after it:

<verdict>
{
  "verdict": "likely_true" | "likely_false" | "misleading" | "unverifiable",
  "confidence": "high" | "medium" | "low",
  "justification": "2-3 sentences explaining the reasoning, referencing what the sources actually said",
  "sources": ["url1", "url2"]
}
</verdict>"""

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}


def extract_claims(article_text: str) -> List[str]:
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=500,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": article_text}],
    )
    raw = _text_of(response.content).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        claims = json.loads(raw)
    except json.JSONDecodeError:
        # JSON array wrapped in other text
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end <= start:
            raise ValueError(f"Could not parse claims from model output:\n{raw}")
        claims = json.loads(raw[start : end + 1])
    if not isinstance(claims, list):
        raise ValueError(f"Expected a JSON array of claims, got: {raw}")
    return [str(c).strip() for c in claims if str(c).strip()][:6]


CLAIM_MAX_TOKENS = 4096  # 2,000 cut off 8% of answers mid-reasoning in the first full run


def build_claim_request(
    claim: str,
    context: Optional[str] = None,
    use_search: bool = True,
    blocked_domains: Optional[List[str]] = None,
    max_searches: Optional[int] = None,
) -> dict:
    """The first Messages API request for one claim, as keyword arguments."""
    user_text = f"Claim to check: {claim}"
    if context:
        user_text += f"\n{context}"

    request = {
        "model": MODEL,
        "max_tokens": CLAIM_MAX_TOKENS,
        "system": VERIFICATION_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_text}],
    }
    if use_search:
        tool = dict(WEB_SEARCH_TOOL)
        if max_searches:
            tool["max_uses"] = int(max_searches)
        if blocked_domains:
            tool["blocked_domains"] = list(blocked_domains)
        request["tools"] = [tool]
    else:
        request["system"] += (
            "\n\nWeb search is not available for this check. Decide from what "
            "you already know and leave sources as an empty list."
        )
    return request


def verify_claim(
    claim: str,
    context: Optional[str] = None,
    use_search: bool = True,
    blocked_domains: Optional[List[str]] = None,
    trace: Optional[dict] = None,
    max_searches: Optional[int] = None,
    first_response=None,
) -> ClaimVerdict:
    """Check one claim. use_search=False gives the closed-book baseline."""
    req = build_claim_request(claim, context, use_search, blocked_domains, max_searches)
    data = _run_agent(
        system=req["system"],
        messages=req["messages"],
        tools=req.get("tools", []),
        dispatch={},  # web_search runs server-side
        max_turns=5,
        max_tokens=req["max_tokens"],
        trace=trace,
        first_response=first_response,
    )
    if data is None:
        return ClaimVerdict(
            claim=claim,
            verdict="unverifiable",
            confidence="low",
            justification="The model did not return a parseable verdict.",
        )
    return ClaimVerdict(
        claim=claim,
        verdict=_pick(data.get("verdict"), VERDICTS, "unverifiable"),
        confidence=_pick(data.get("confidence"), CONFIDENCES, "low"),
        justification=str(data.get("justification", "")),
        sources=_as_str_list(data.get("sources")),
    )


def check_article(article_text: str) -> List[ClaimVerdict]:
    return [verify_claim(c) for c in extract_claims(article_text)]


# Check 2: image provenance and manipulation


def _host_path(url: str) -> str:
    from urllib.parse import urlparse

    p = urlparse(url if "://" in (url or "") else "http://" + (url or ""))
    host = p.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return f"{host}{p.path}".lower()


def _blocked(url: str, entries: List[str]) -> bool:
    """Same matching rules as the web search tool's blocked_domains."""
    hp = _host_path(url)
    host, _, path = hp.partition("/")
    for entry in entries:
        e_host, _, e_path = entry.lower().partition("/")
        if not (host == e_host or host.endswith("." + e_host)):
            continue
        if not e_path or re.match("^" + re.escape(e_path).replace(r"\*", "[^/]*"), path):
            return True
    return False


def reverse_image_search(
    image_url: Optional[str], num_results: int = 8, blocked_domains: Optional[List[str]] = None
) -> dict:
    """Find earlier appearances of an image via Google Lens."""
    if not SERPAPI_KEY:
        return {"error": "SERPAPI_API_KEY is not set, cannot run reverse image search"}
    if not image_url:
        return {"error": "no public image URL was supplied, cannot run reverse image search"}
    try:
        response = requests.get(
            "https://serpapi.com/search",
            params={"engine": "google_lens", "url": image_url, "api_key": SERPAPI_KEY},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"error": f"reverse image search request failed: {exc}"}
    if data.get("error"):
        return {"error": f"SerpApi: {data['error']}"}
    matches = data.get("visual_matches", [])
    removed = 0
    if blocked_domains:
        kept = [m for m in matches if not _blocked(m.get("link") or "", blocked_domains)]
        removed = len(matches) - len(kept)
        matches = kept
    result = {
        "matches": [
            {"title": m.get("title"), "source": m.get("source"), "link": m.get("link")}
            for m in matches[:num_results]
        ]
    }
    if removed:
        result["note"] = f"{removed} result(s) from fact-checking sites were withheld"
    return result


_manipulation_pipeline = None
_manipulation_lock = threading.Lock()

# Note shown to the model with the classifier score (from the stage 2 dev set).
CLASSIFIER_NOTE = (
    "This classifier was trained on images of human faces and over-scores most "
    "other photographs, so its score separates real from synthetic images only "
    "weakly. On a development set of news-type images the median Deepfake score "
    "was 0.64 for genuine photographs (most between 0.55 and 0.75), 0.65 for "
    "edited images and 0.77 for AI-generated images, about a third of which "
    "scored above 0.80. A higher score makes synthesis somewhat more likely; "
    "weigh it together with visible signs of generation or editing and with the "
    "search results, and do not dismiss it just because it falls in the range "
    "genuine photographs can reach. It says nothing about whether a caption is true."
)


def classify_manipulation(image_path: str) -> dict:
    """Run the local deepfake classifier on an image."""
    global _manipulation_pipeline
    if not os.path.exists(image_path):
        return {"error": f"local image file not found: {image_path}"}
    # load once under a lock (parallel transformers imports can fail)
    with _manipulation_lock:
        if _manipulation_pipeline is None:
            try:
                from transformers import pipeline
            except ImportError:
                return {"error": "transformers is not installed; run `pip install transformers torch pillow`"}
            _manipulation_pipeline = pipeline("image-classification", model=MANIPULATION_MODEL_ID)
    with _manipulation_lock:
        predictions = _manipulation_pipeline(image_path)
    return {
        "predictions": [
            {"label": str(p.get("label")), "score": round(float(p.get("score", 0.0)), 4)}
            for p in predictions
        ]
    }


def _classifier_for_model(result: dict) -> dict:
    """The classifier output as shown to the model: raw score plus a note."""
    if "error" in result:
        return result
    score = None
    for p in result.get("predictions", []):
        if "fake" in str(p.get("label", "")).lower():
            score = float(p["score"])
    if score is None:  # label names differ: fall back to the top prediction
        top = max(result.get("predictions", []), key=lambda p: p.get("score", 0), default={})
        return {"top_label": top.get("label", ""), "score": top.get("score", 0.0), "calibration_note": CLASSIFIER_NOTE}
    return {"deepfake_score": round(score, 4), "calibration_note": CLASSIFIER_NOTE}


IMAGE_TOOLS = [
    {
        "name": "reverse_image_search",
        "description": (
            "Search the web via Google Lens for earlier appearances of the "
            "image you have been given. Returns matched pages with their "
            "titles, sources and links. Use this to work out where and when "
            "the image first appeared and what claims have been attached to "
            "it elsewhere, which is how a miscontextualised (but otherwise "
            "untouched) photo gets caught."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "classify_manipulation",
        "description": (
            "Run a local classifier that estimates whether the image itself "
            "has been synthetically generated or manipulated, as opposed to "
            "being an authentic, unaltered photograph. Returns a Deepfake "
            "score between 0 and 1 with a note on how to read it. Treat this "
            "as one weak signal among several, since this classifier degrades "
            "sharply on images unlike its training data."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

IMAGE_VERIFICATION_SYSTEM_PROMPT = """You are a careful visual fact-checker.
You will be given an image, together with any caption or claim it was
published alongside. Some images carry their own claim as text printed
directly on the image; read that off the image yourself before deciding
what needs checking, and treat it as the claim under test if no separate
caption was supplied.

You have two tools:
- reverse_image_search finds earlier appearances of the image on the web,
  which is how you catch a real photo being reused with a false date,
  location or context attached.
- classify_manipulation estimates whether the image's pixels have been
  synthesised or altered, which is how you catch a fabricated image rather
  than a misused real one.

Process:
- Look at the image yourself first. Note what it appears to show and
  transcribe any text overlaid on it.
- Call both tools before concluding, unless one is clearly irrelevant, and
  request them together in the same step where you can, since neither
  depends on the other.
  Reverse image search results and the manipulation classifier answer
  different questions, and a real photo can still be used to tell a false
  story, so a clean classifier result does not settle the case on its own.
- If the classifier and the search results point in different directions,
  say so rather than picking whichever is more convenient.
- The user message gives today's date. Events dated before it may have
  happened after your training data ends, so do not treat a recent date as
  impossible or false just because you have no record of the event.

Choosing between authentic, miscontextualised and unverifiable:
- Genuine news photographs often have no earlier copy online, or only
  loosely related matches such as other photos of similar events. The
  absence of a match confirming the caption is not evidence against it.
- Call an image "miscontextualised" only when you have positive evidence
  that it belongs to a different date, place, person or event: an earlier
  appearance with a different context, or visible detail that contradicts
  the caption (for example a caption describing snow on a photo with none).
- Check each specific detail the caption or printed claim asserts (the
  place or landmark, the event, the year, the season or weather, colour or
  black-and-white, the number of people, any signs or text) against what
  you can see. A clear mismatch is positive evidence of miscontextualisation
  even when reverse image search finds nothing.
- If there is a caption or printed claim, nothing you can see or find
  contradicts it, its details fit the scene, and there is no real sign of
  manipulation, prefer "authentic_truthful" with medium or low confidence
  over "unverifiable".
- If there is no caption and no printed claim, the question is only whether
  the image itself is genuine. Look closely for signs of generation or
  editing and weigh the classifier score; do not call the image authentic
  merely because there is nothing to contradict.
- Keep "unverifiable" for cases where the evidence genuinely conflicts, or
  where the caption's key claim (such as who a person is) can neither be
  checked against the image nor found in the search results.

If a source in the search results describes the image as doctored,
photoshopped, altered, fake or AI-generated, or the image combines elements
from different photographs, the category is "content_manipulation", even
when the pieces are real photographs and even when the image is also being
shared with a false story.

Classify the image into exactly one of these categories:
- "content_manipulation": the image itself appears synthesised, generated
  or digitally altered.
- "miscontextualised": the image is authentic but is being presented with
  a false date, location, identity or circumstance.
- "authentic_truthful": the image is authentic and the claim attached to it
  checks out.
- "unverifiable": the available evidence does not support a confident call
  either way.

The "verdict" field judges the claim or caption attached to the asset, and
uses the same scale as the text checks: "likely_true", "likely_false",
"misleading" (partly accurate but leaving a false overall impression, for
example a genuine photo from a real event captioned with the wrong year),
or "unverifiable".

When you are done, end your response with a JSON block in exactly this
format, and nothing after it:

<verdict>
{
  "category": "content_manipulation" | "miscontextualised" | "authentic_truthful" | "unverifiable",
  "verdict": "likely_true" | "likely_false" | "misleading" | "unverifiable",
  "confidence": "high" | "medium" | "low",
  "justification": "2-3 sentences explaining the reasoning, referencing what the tools actually returned",
  "manipulation_label": "Deepfake if the classifier returned a Deepfake score, otherwise empty string",
  "manipulation_score": 0.0,
  "sources": ["url1", "url2"]
}
</verdict>"""


def _encode_image(image_path: str) -> tuple:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")
    size = os.path.getsize(image_path)
    if size > MAX_IMAGE_BYTES:
        raise ValueError(
            f"{image_path} is {size / 1_048_576:.1f} MB; the API limit is 5 MB, so resize it first"
        )
    media_type, _ = mimetypes.guess_type(image_path)
    if media_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        media_type = "image/jpeg"
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return media_type, data


NO_IMAGE_NOTE = """

For this check the image itself is not shown to you. Work only from the
caption and the tool results, and do not guess at what the image shows or
at any text printed on it."""


def verify_image(
    image_path: str,
    image_url: Optional[str] = None,
    caption: Optional[str] = None,
    *,
    trace: Optional[dict] = None,
    blocked_domains: Optional[List[str]] = None,
    send_image: bool = True,
    model_image_path: Optional[str] = None,
) -> ImageVerdict:
    """Check an image. image_url is used for reverse search; caption is the claim posted with it."""
    user_text = f"Today's date is {_dt.date.today().isoformat()}. "
    user_text += "Here is the image to check." if send_image else "An image is being checked."
    if caption:
        user_text += f" It was published with this caption or claim: {caption!r}"
    else:
        user_text += " No separate caption was supplied; read any text printed on the image yourself."
    if not image_url:
        user_text += (
            " No public URL is available for this image, so reverse_image_search "
            "will not be able to run; rely on the manipulation classifier and your "
            "own reading of the image instead."
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")
    content: List[dict] = []
    if send_image:
        media_type, image_data = _encode_image(model_image_path or image_path)
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}})
    else:
        user_text = user_text.replace(" read any text printed on the image yourself.", " work from the tool results.")
    content.append({"type": "text", "text": user_text})

    def _search(_inp):
        result = reverse_image_search(image_url, blocked_domains=blocked_domains)
        if trace is not None:
            if "matches" in result:
                trace["serpapi_searches"] = trace.get("serpapi_searches", 0) + 1
                for m in result["matches"]:
                    if m.get("link") and m["link"] not in trace["retrieved_urls"]:
                        trace["retrieved_urls"].append(m["link"])
            trace.setdefault("tool_results", []).append({"tool": "reverse_image_search", **result})
        return result

    def _classify(_inp):
        result = classify_manipulation(image_path)
        if trace is not None:
            trace.setdefault("tool_results", []).append({"tool": "classify_manipulation", **result})
        return _classifier_for_model(result)

    data = _run_agent(
        system=IMAGE_VERIFICATION_SYSTEM_PROMPT + ("" if send_image else NO_IMAGE_NOTE),
        messages=[{"role": "user", "content": content}],
        tools=IMAGE_TOOLS,
        dispatch={"reverse_image_search": _search, "classify_manipulation": _classify},
        max_turns=6,
        max_tokens=2000,
        trace=trace,
    )
    ref = image_url or image_path
    if data is None:
        return ImageVerdict(
            image_ref=ref,
            category="unverifiable",
            verdict="unverifiable",
            confidence="low",
            justification="The model did not return a parseable verdict.",
        )
    return ImageVerdict(
        image_ref=ref,
        category=_pick(data.get("category"), CATEGORIES, "unverifiable"),
        verdict=_pick(data.get("verdict"), VERDICTS, "unverifiable"),
        confidence=_pick(data.get("confidence"), CONFIDENCES, "low"),
        justification=str(data.get("justification", "")),
        manipulation_label=str(data.get("manipulation_label") or ""),
        manipulation_score=_as_float(data.get("manipulation_score")),
        sources=_as_str_list(data.get("sources")),
    )


# Check 3: C2PA content credentials

try:
    import c2pa
except ImportError:
    c2pa = None


class ManifestUnavailable(Exception):
    """c2pa-python is not installed."""


_NO_MANIFEST_MARKERS = ("manifestnotfound", "manifest not found", "no jumbf", "no claim")


def read_c2pa_manifest(file_path: str) -> Optional[dict]:
    """Return the active C2PA manifest and validation results, or None if there is none."""
    if c2pa is None:
        raise ManifestUnavailable("c2pa-python is not installed; run `pip install c2pa-python`")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"file not found: {file_path}")

    def _safe(call):
        try:
            return call()
        except Exception:
            return None

    try:
        with c2pa.Reader(file_path) as reader:
            try:
                active = reader.get_active_manifest()
            except KeyError:
                active = None  # manifest store present but no active manifest
            if not active:
                return None
            return {
                "manifest": active,
                "validation_state": _safe(reader.get_validation_state),
                "validation_results": _safe(reader.get_validation_results),
            }
    except Exception as exc:
        if any(m in str(exc).lower() for m in _NO_MANIFEST_MARKERS):
            return None
        raise


def summarise_manifest(raw: dict) -> dict:
    """Reduce a manifest to signer, actions, ingredients and validation."""
    manifest = raw.get("manifest") or {}
    signature_info = manifest.get("signature_info") or {}

    actions = []
    for assertion in manifest.get("assertions", []) or []:
        if str(assertion.get("label", "")).startswith("c2pa.actions"):
            for action in (assertion.get("data") or {}).get("actions", []) or []:
                name = action.get("action")
                if name:
                    entry = {"action": name}
                    if action.get("digitalSourceType"):
                        entry["digital_source_type"] = action["digitalSourceType"]
                    if action.get("softwareAgent"):
                        entry["software_agent"] = action["softwareAgent"]
                    actions.append(entry)

    ingredients = [i.get("title") for i in manifest.get("ingredients", []) or [] if i.get("title")]

    return {
        "claim_generator": manifest.get("claim_generator")
        or ", ".join(str(g.get("name")) for g in manifest.get("claim_generator_info", []) or [] if g.get("name")),
        "title": manifest.get("title"),
        "issuer": signature_info.get("issuer"),
        "signing_time": signature_info.get("time"),
        "actions": actions,
        "ingredients": ingredients,
        "validation_state": raw.get("validation_state"),
        "validation_results": compact_validation(raw.get("validation_results")),
    }


def compact_validation(results) -> dict:
    """Keep only the failure codes and the number of passed checks."""
    if not isinstance(results, dict):
        return {"raw": results} if results else {}
    active = results.get("activeManifest") or {}
    failures = [
        {"code": f.get("code"), "explanation": f.get("explanation")}
        for f in active.get("failure", []) or []
    ]
    return {
        "failures": failures,
        "passed_checks": len(active.get("success", []) or []),
        "informational": [i.get("code") for i in active.get("informational", []) or []],
        "ingredient_failures": sum(
            len((d or {}).get("failure", []) or []) for d in results.get("ingredientDeltas", []) or []
            if isinstance(d, dict)
        ),
    }


def check_c2pa_manifest(file_path: str) -> dict:
    """Tool entry point. Returns manifest_present False when there is no manifest."""
    try:
        raw = read_c2pa_manifest(file_path)
    except (ManifestUnavailable, FileNotFoundError) as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"could not read C2PA data from {file_path}: {exc}"}
    if raw is None:
        return {"manifest_present": False}
    return {"manifest_present": True, **summarise_manifest(raw)}


PROVENANCE_TOOLS = [
    {
        "name": "check_c2pa_manifest",
        "description": (
            "Read the C2PA content credentials manifest attached to the media "
            "file under review, if one exists. Returns whether a manifest is "
            "present and, if so, who generated and signed it, what actions "
            "(edits, AI generation and so on) it records, what earlier assets "
            "it was built from, and whether it passed validation. A file with "
            "no manifest returns manifest_present: false, which is common and "
            "not itself suspicious."
        ),
        # no file_path argument, so only the file under review can be read
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
]

PROVENANCE_SYSTEM_PROMPT = """You are a careful provenance checker. You
will be given an image or video under review, together with any caption or
claim it was published alongside, and optionally the findings of an earlier
reverse-image-search or manipulation-classifier check on the same asset.

You have one tool, check_c2pa_manifest, which reads the asset's C2PA
content credentials (if any) and reports what they say.

Process:
- Call check_c2pa_manifest first, even if you suspect the answer will be
  "no manifest": the absence of a manifest is itself evidence, not a dead
  end, and you should not guess at this without checking.
- No manifest at all is common for genuine photos too, since most cameras
  and phones do not attach one yet. Treat a missing manifest as
  inconclusive on its own, never as proof of manipulation by itself.
- A manifest whose validation_state or validation_results indicate a
  failure (for example a hash mismatch between the recorded and actual
  asset) is meaningfully stronger evidence that the file was altered after
  the credentials were attached, especially if the claim is that the asset
  is untouched.
- Pay attention to what the actions list records (for example a
  digital_source_type indicating generative AI) and to the ingredients list,
  which shows what source assets the manifest says this file was built from.
- If you were given prior reverse-image-search or classifier findings,
  reconcile them with what the manifest says rather than ignoring one in
  favour of the other. Say so explicitly if they point in different
  directions.

Classify the asset into exactly one of these categories:
- "content_manipulation": the manifest (or its absence combined with other
  evidence) indicates the asset itself was synthesised or altered.
- "miscontextualised": the asset's provenance trail looks intact but the
  claim attached to it does not match what the manifest or ingredients
  describe.
- "authentic_truthful": the manifest validates cleanly and is consistent
  with the claim, or no manifest exists and nothing else raises concern.
- "unverifiable": the available evidence does not support a confident call
  either way.

The "verdict" field judges the claim or caption attached to the asset, and
uses the same scale as the text checks: "likely_true", "likely_false",
"misleading" (partly accurate but leaving a false overall impression, for
example a genuine photo from a real event captioned with the wrong year),
or "unverifiable".

Once you have called the tool and have enough to decide, your entire final
reply must be the JSON block below and nothing else: no summary, no bullet
points, no explanation outside the "justification" field.

<verdict>
{
  "category": "content_manipulation" | "miscontextualised" | "authentic_truthful" | "unverifiable",
  "verdict": "likely_true" | "likely_false" | "misleading" | "unverifiable",
  "confidence": "high" | "medium" | "low",
  "justification": "2-3 sentences explaining the reasoning, referencing what the manifest actually said",
  "manifest_present": true or false,
  "validation_state": "whatever the tool reported, or empty string",
  "issuer": "the signing issuer the tool reported, or empty string",
  "sources": []
}
</verdict>"""


def verify_provenance(
    file_path: str,
    caption: Optional[str] = None,
    prior_findings: Optional[str] = None,
    *,
    trace: Optional[dict] = None,
    display_name: Optional[str] = None,
) -> ProvenanceVerdict:
    """Check C2PA provenance. display_name hides the real file name from the model."""
    # record manifest facts from the tool, not from the model's reply
    tool_result = check_c2pa_manifest(file_path)
    if trace is not None:
        trace["tool_results"] = [{"tool": "check_c2pa_manifest", **tool_result}]

    user_text = f"File under review: {display_name or os.path.basename(file_path)}"
    if caption:
        user_text += f"\nIt was published with this caption or claim: {caption!r}"
    if prior_findings:
        user_text += f"\nEarlier findings on this same asset: {prior_findings}"

    data = _run_agent(
        system=PROVENANCE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_text}],
        tools=PROVENANCE_TOOLS,
        dispatch={"check_c2pa_manifest": lambda _inp: tool_result},
        max_turns=4,
        max_tokens=1200,
        trace=trace,
    )

    present = bool(tool_result.get("manifest_present", False))
    state = str(tool_result.get("validation_state") or "")
    issuer = str(tool_result.get("issuer") or "")

    if data is None:
        return ProvenanceVerdict(
            file_ref=file_path,
            category="unverifiable",
            verdict="unverifiable",
            confidence="low",
            justification="The model did not return a parseable verdict.",
            manifest_present=present,
            validation_state=state,
            issuer=issuer,
        )
    return ProvenanceVerdict(
        file_ref=file_path,
        category=_pick(data.get("category"), CATEGORIES, "unverifiable"),
        verdict=_pick(data.get("verdict"), VERDICTS, "unverifiable"),
        confidence=_pick(data.get("confidence"), CONFIDENCES, "low"),
        justification=str(data.get("justification", "")),
        manifest_present=present,
        validation_state=state,
        issuer=issuer,
        sources=_as_str_list(data.get("sources")),
    )


# Orchestration


def _summarise_image_finding(v: ImageVerdict) -> str:
    parts = [f"category={v.category}", f"verdict={v.verdict}", f"confidence={v.confidence}"]
    if v.manipulation_label:
        parts.append(f"classifier={v.manipulation_label} ({v.manipulation_score:.2f})")
    return "; ".join(parts) + f". {v.justification}"


def check_item(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    caption: Optional[str] = None,
    run_provenance: bool = True,
) -> Report:
    """Run every check the inputs allow and collect the results."""
    if not text and not image_path:
        raise ValueError("Provide text, an image path, or both.")

    get_client()  # fail fast on a missing API key instead of logging it three times

    report = Report()

    if text:
        try:
            claims = extract_claims(text)
        except Exception as exc:
            report.errors.append(f"claim extraction failed: {exc}")
            claims = []
        for claim in claims:
            try:
                report.claims.append(verify_claim(claim))
            except Exception as exc:
                report.errors.append(f"claim check failed for {claim!r}: {exc}")

    if image_path:
        try:
            report.image = verify_image(image_path, image_url=image_url, caption=caption)
        except Exception as exc:
            report.errors.append(f"image check failed: {exc}")

        if run_provenance:
            prior = _summarise_image_finding(report.image) if report.image else None
            try:
                report.provenance = verify_provenance(
                    image_path, caption=caption, prior_findings=prior
                )
            except Exception as exc:
                report.errors.append(f"provenance check failed: {exc}")

    return report


def print_report(report: Report) -> None:
    line = "-" * 60
    if report.claims:
        print("\nTEXT CLAIMS")
        print(line)
        for c in report.claims:
            print(f"CLAIM: {c.claim}")
            print(f"VERDICT: {c.verdict} (confidence: {c.confidence})")
            print(f"JUSTIFICATION: {c.justification}")
            print(f"SOURCES: {', '.join(c.sources) or 'none'}")
            print(line)
    if report.image:
        v = report.image
        print("\nIMAGE CHECK")
        print(line)
        print(f"CATEGORY: {v.category}")
        print(f"VERDICT: {v.verdict} (confidence: {v.confidence})")
        print(f"JUSTIFICATION: {v.justification}")
        print(f"CLASSIFIER: {v.manipulation_label or 'n/a'} ({v.manipulation_score:.2f})")
        print(f"SOURCES: {', '.join(v.sources) or 'none'}")
    if report.provenance:
        p = report.provenance
        print("\nC2PA PROVENANCE")
        print(line)
        print(f"CATEGORY: {p.category}")
        print(f"VERDICT: {p.verdict} (confidence: {p.confidence})")
        print(f"JUSTIFICATION: {p.justification}")
        print(f"MANIFEST PRESENT: {p.manifest_present}")
        print(f"VALIDATION STATE: {p.validation_state or 'n/a'}")
        print(f"ISSUER: {p.issuer or 'n/a'}")
    if report.errors:
        print("\nERRORS")
        print(line)
        for e in report.errors:
            print(f" - {e}")


def _prompt_for_inputs() -> tuple:
    """Ask for each input in the terminal."""
    print("No arguments given, so I'll ask for them. Press Enter to skip any prompt.\n")
    text = input("Text to check: ").strip() or None
    image = input("Local image path: ").strip().strip('"').strip("'") or None
    image_url = caption = None
    if image:
        if not os.path.isfile(image):
            print(f"Can't find {image}, so the image checks will be skipped.")
            image = None
        else:
            image_url = input("Public URL of the same image (for reverse search): ").strip() or None
            caption = input("Caption or claim published with the image: ").strip() or None
    print()
    return text, image, image_url, caption


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check text claims and images for misinformation.")
    parser.add_argument("--text", help="Text to extract and check claims from.")
    parser.add_argument("--text-file", help="Read the text from a file instead.")
    parser.add_argument("--image", help="Local path to an image to check.")
    parser.add_argument("--image-url", help="Public URL of the same image, for reverse image search.")
    parser.add_argument("--caption", help="Caption or claim the image was published with.")
    parser.add_argument("--no-provenance", action="store_true", help="Skip the C2PA check.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log tool activity.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    text = args.text
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()

    if not text and not args.image:
        if not sys.stdin.isatty():
            parser.error("give --text, --text-file or --image (or a combination)")
        # no arguments (e.g. VS Code Run button): ask for inputs
        text, args.image, args.image_url, args.caption = _prompt_for_inputs()
        if not text and not args.image:
            print("Nothing to check. Enter some text or an image path next time.")
            return 0

    try:
        report = check_item(
            text=text,
            image_path=args.image,
            image_url=args.image_url,
            caption=args.caption,
            run_provenance=not args.no_provenance,
        )
    except RuntimeError as exc:  # missing API key and similar setup problems
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print_report(report)
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())

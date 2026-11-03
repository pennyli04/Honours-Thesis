"""Offline tests for factcheck.py using a fake API client.

    python -m pytest -q test_factcheck.py
"""

import json
import re
import os
from types import SimpleNamespace as NS

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test")
import factcheck as fc  # noqa: E402


def text(t):
    return NS(type="text", text=t)


def tool_use(name, id_="tu_1"):
    return NS(type="tool_use", name=name, id=id_, input={})


def resp(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop)


def verdict(**kw):
    base = {"verdict": "likely_true", "confidence": "high", "justification": "ok {braces} fine", "sources": ["https://abs.gov.au"]}
    base.update(kw)
    return f"Reasoning...\n<verdict>\n{json.dumps(base)}\n</verdict>"


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        msgs = json.loads(json.dumps(kw["messages"], default=lambda o: vars(o)))
        _check_well_formed(msgs)
        self.calls.append({**kw, "messages": msgs})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _check_well_formed(msgs):
    assert msgs[0]["role"] == "user"
    for a, b in zip(msgs, msgs[1:]):
        assert a["role"] != b["role"], "roles must alternate"
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            continue
        ids = {b["id"] for b in m["content"] if b.get("type") == "tool_use"}
        if ids:
            nxt = msgs[i + 1]["content"]
            got = {b["tool_use_id"] for b in nxt if b.get("type") == "tool_result"}
            assert ids == got, "every tool_use needs a tool_result"


@pytest.fixture
def fake(monkeypatch):
    def install(script):
        client = FakeClient(script)
        monkeypatch.setattr(fc, "_client", client)
        return client
    return install


@pytest.fixture
def jpg(tmp_path):
    from PIL import Image
    p = tmp_path / "test.jpg"
    Image.new("RGB", (32, 32), "blue").save(p)
    return str(p)


# parsing

def test_parse_verdict_handles_braces_and_fences():
    t = '<verdict>```json\n{"verdict":"likely_false","justification":"a {b} c"}\n```</verdict>'
    assert fc._parse_verdict(t)["justification"] == "a {b} c"
    assert fc._parse_verdict("no block") is None
    assert fc._parse_verdict("<verdict>not json</verdict>") is None


def test_bad_enum_values_are_normalised(fake):
    fake([resp(text(verdict(verdict="TRUE!!", confidence="High")))])
    v = fc.verify_claim("x")
    assert v.verdict == "unverifiable" and v.confidence == "high"


# check 1

def test_extract_claims_salvages_prose(fake):
    fake([resp(text('Here you go:\n["Claim A", "Claim B", ""]'))])
    assert fc.extract_claims("article") == ["Claim A", "Claim B"]


def test_verify_claim_resumes_after_pause_turn(fake):
    c = fake([resp(text("searching"), stop="pause_turn"), resp(text(verdict()))])
    v = fc.verify_claim("census date")
    assert v.verdict == "likely_true" and v.sources == ["https://abs.gov.au"]
    assert len(c.calls) == 2


def test_verify_claim_nudges_when_verdict_missing(fake):
    c = fake([resp(text("I think it is true.")), resp(text(verdict(verdict="likely_false")))])
    assert fc.verify_claim("x").verdict == "likely_false"
    assert c.calls[-1]["tool_choice"] == {"type": "none"}


def test_verify_claim_gives_unverifiable_when_model_never_complies(fake):
    fake([resp(text("nope")), resp(text("still nope"))])
    v = fc.verify_claim("x")
    assert v.verdict == "unverifiable" and v.confidence == "low"


# check 2

def test_verify_image_runs_tools_and_survives_tool_crash(fake, jpg, monkeypatch):
    def boom(_):
        raise RuntimeError("model download failed")
    monkeypatch.setattr(fc, "classify_manipulation", boom)
    c = fake([
        resp(tool_use("reverse_image_search", "a"), tool_use("classify_manipulation", "b"), stop="tool_use"),
        resp(text(verdict(category="miscontextualised", verdict="likely_false", manipulation_score="0.12"))),
    ])
    v = fc.verify_image(jpg, caption="Floods 2026")
    assert v.category == "miscontextualised" and v.manipulation_score == 0.12
    results = c.calls[1]["messages"][-1]["content"]
    assert "SERPAPI" in results[0]["content"] or "URL" in results[0]["content"]
    assert "model download failed" in results[1]["content"]


def test_turn_cap_on_tool_use_still_gets_verdict(fake, jpg, monkeypatch):
    monkeypatch.setattr(fc, "classify_manipulation", lambda p: {"predictions": []})
    script = [resp(tool_use("classify_manipulation", f"t{i}"), stop="tool_use") for i in range(6)]
    c = fake(script + [resp(text(verdict(category="unverifiable")))])
    v = fc.verify_image(jpg)
    assert v.category == "unverifiable"
    last_user = c.calls[-1]["messages"][-1]["content"]
    assert last_user[0]["type"] == "tool_result" and last_user[-1]["type"] == "text"


def test_oversized_image_rejected(tmp_path):
    big = tmp_path / "big.jpg"
    big.write_bytes(b"\0" * (fc.MAX_IMAGE_BYTES + 1))
    with pytest.raises(ValueError):
        fc._encode_image(str(big))


# check 3

def test_c2pa_plain_jpeg_has_no_manifest(jpg):
    assert fc.check_c2pa_manifest(jpg) == {"manifest_present": False}


def test_c2pa_missing_file_is_error(tmp_path):
    assert "error" in fc.check_c2pa_manifest(str(tmp_path / "nope.jpg"))


def test_summarise_manifest_extracts_ai_actions():
    raw = {"manifest": {
        "claim_generator": "Firefly", "signature_info": {"issuer": "Adobe Inc.", "time": "2026"},
        "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [
            {"action": "c2pa.created", "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"}]}}],
        "ingredients": [{"title": "base.jpg"}]}, "validation_state": "Valid"}
    s = fc.summarise_manifest(raw)
    assert s["issuer"] == "Adobe Inc." and s["ingredients"] == ["base.jpg"]
    assert "trainedAlgorithmicMedia" in s["actions"][0]["digital_source_type"]


def test_provenance_uses_tool_facts_not_model_echo(fake, jpg):
    fake([
        resp(tool_use("check_c2pa_manifest"), stop="tool_use"),
        resp(text(verdict(category="authentic_truthful", manifest_present=True, issuer="Made Up"))),
    ])
    p = fc.verify_provenance(jpg)
    assert p.manifest_present is False and p.issuer == ""


# orchestration

def test_check_item_full_run_and_json(fake, jpg, monkeypatch):
    monkeypatch.setattr(fc, "classify_manipulation", lambda p: {"predictions": [{"label": "Realism", "score": 0.9}]})
    c = fake([
        resp(text('["The census was held on 11 August 2026"]')),
        resp(text(verdict())),
        resp(tool_use("classify_manipulation"), stop="tool_use"),
        resp(text(verdict(category="authentic_truthful", manipulation_label="Realism", manipulation_score=0.9))),
        resp(tool_use("check_c2pa_manifest"), stop="tool_use"),
        resp(text(verdict(category="authentic_truthful"))),
    ])
    r = fc.check_item(text="article", image_path=jpg)
    assert len(r.claims) == 1 and r.image and r.provenance and not r.errors
    assert "classifier=Realism" in c.calls[4]["messages"][0]["content"]
    json.dumps(r.to_dict())


def test_check_item_isolates_failures(fake, jpg):
    fake([RuntimeError("API down"), RuntimeError("API down"), RuntimeError("API down")])
    r = fc.check_item(text="article", image_path=jpg)
    assert len(r.errors) == 3 and r.image is None and r.provenance is None


def test_cli_requires_input():
    with pytest.raises(SystemExit):
        fc.main([])


# evaluation support

import eval_averitec as ev  # noqa: E402


def test_trace_records_usage_and_urls(fake):
    r = resp(
        NS(type="web_search_tool_result", content=[NS(url="https://a.org/1"), NS(url="https://a.org/1")]),
        NS(type="web_search_tool_result", content=NS(type="web_search_tool_result_error", error_code="max_uses_exceeded")),
        text(verdict()),
    )
    r.usage = NS(input_tokens=100, output_tokens=20, server_tool_use=NS(web_search_requests=2))
    c = fake([r])
    t = fc.new_trace()
    fc.verify_claim("x", context="Speaker: A", blocked_domains=["politifact.com"], trace=t)
    assert t["retrieved_urls"] == ["https://a.org/1"] and t["web_search_requests"] == 2
    assert c.calls[0]["tools"][0]["blocked_domains"] == ["politifact.com"]
    assert "Speaker: A" in c.calls[0]["messages"][0]["content"]


def test_no_search_sends_no_tools(fake):
    c = fake([resp(text("no block")), resp(text(verdict()))])
    fc.verify_claim("x", use_search=False)
    assert all("tools" not in call and "tool_choice" not in call for call in c.calls)


def test_blocklist_is_valid_for_the_api():
    import re as _re
    assert len(ev.BLOCKED_DOMAINS) < 64 and len(set(ev.BLOCKED_DOMAINS)) == len(ev.BLOCKED_DOMAINS)
    for e in ev.BLOCKED_DOMAINS:
        host = e.split("/")[0]
        assert "://" not in e and e.isascii() and "*" not in host
        assert _re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", host), e


def test_leakage_matching():
    item = {"fact_checking_article": "https://web.archive.org/web/2020/https://www.politifact.com/factchecks/2020/x/"}
    f = ev.leakage_flags(item, ["http://politifact.com/factchecks/2020/x"], [])
    assert f["leak_exact"] and f["leak_factcheck_site"]
    f = ev.leakage_flags(item, ["https://www.usatoday.com/story/news/politics/2020/"], [])
    assert not f["leak_exact"] and not f["leak_factcheck_site"]
    assert ev._matches_block("https://www.usatoday.com/story/news/factcheck/2020/10/1/x", "usatoday.com/story/news/factcheck")
    assert ev._matches_block("https://www.washingtonpost.com/politics/2020/10/fact-checker/x", "washingtonpost.com/politics/2020/*/fact-checker")


def test_sampling_is_stratified_and_repeatable():
    data = [{"_id": i, "label": "Refuted" if i % 4 else "Supported"} for i in range(400)]
    a, b = ev.stratified_sample(data, 40, 7), ev.stratified_sample(data, 40, 7)
    assert [x["_id"] for x in a] == [x["_id"] for x in b] and len(a) == 40
    assert sum(x["label"] == "Supported" for x in a) == 10


def test_mcnemar():
    assert ev._mcnemar_exact(0, 0) == 1.0
    assert abs(ev._mcnemar_exact(2, 10) - 0.0386) < 0.001


def test_derived_leak_flags():
    r = ev.add_derived_flags({"trace": {"retrieved_urls": ["https://cnn.com/2020/10/26/politics/x-fact-check/index.html"]},
                              "justification": "Africa Check found the figure was 45%."})
    assert r["factcheck_page_retrieved"] and r["cites_factchecker"]
    r = ev.add_derived_flags({"trace": {"retrieved_urls": ["https://saps.gov.za/report"]}, "justification": "SAPS data."})
    assert not r["factcheck_page_retrieved"] and not r["cites_factchecker"]


def test_run_exclude(tmp_path, monkeypatch, fake):
    import json as _j
    data = [{"claim": f"c{i}", "label": "Refuted"} for i in range(10)]
    (tmp_path / "d.json").write_text(_j.dumps(data))
    (tmp_path / "pilot.jsonl").write_text("\n".join(_j.dumps({"id": i}) for i in range(8)))
    monkeypatch.chdir(tmp_path)
    fake([resp(text(verdict(verdict="likely_false"))) for _ in range(2)])
    ev.main(["run", "--data", "d.json", "--exclude", "pilot.jsonl", "--workers", "1", "--yes"])
    ids = [_j.loads(l)["id"] for l in open("results/averitec_search_blocked.jsonl")]
    assert sorted(ids) == [8, 9]


def test_misleading_verdict_and_prompt_rules(fake):
    c = fake([resp(text(verdict(verdict="misleading")))])
    assert fc.verify_claim("Russia sent aid to the US").verdict == "misleading"
    system = c.calls[0]["system"]
    assert '"misleading"' in system and "X claimed that Y" in system
    assert '"misleading"' in fc.IMAGE_VERIFICATION_SYSTEM_PROMPT and '"misleading"' in fc.PROVENANCE_SYSTEM_PROMPT


def test_rescoring_uses_current_mapping():
    r = ev.add_derived_flags({"gold_label": "Conflicting Evidence/Cherrypicking", "gold_mapped": "unverifiable",
                              "pred": "misleading", "correct": False, "trace": {}})
    assert r["gold_mapped"] == "misleading" and r["correct"] is True


def test_run_refuses_to_mix_prompt_versions(tmp_path, monkeypatch):
    import json as _j
    (tmp_path / "d.json").write_text(_j.dumps([{"claim": "c", "label": "Refuted"}]))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "averitec_search_blocked.jsonl").write_text(_j.dumps({"id": 0, "prompt_version": "v1"}) + "\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        ev.main(["run", "--data", "d.json"])
    assert "prompt version" in str(e.value)


def test_run_stops_on_credit_error(tmp_path, monkeypatch, fake):
    import json as _j
    (tmp_path / "d.json").write_text(_j.dumps([{"claim": f"c{i}", "label": "Refuted"} for i in range(5)]))
    monkeypatch.chdir(tmp_path)
    fake([resp(text(verdict(verdict="likely_false")))] + [RuntimeError("Your credit balance is too low")] * 10)
    with pytest.raises(SystemExit) as e:
        ev.main(["run", "--data", "d.json", "--workers", "1", "--yes"])
    assert "credit balance" in str(e.value)
    lines = open("results/averitec_search_blocked.jsonl").read().splitlines()
    assert len(lines) == 1 and _j.loads(lines[0])["error"] is None


def test_v3_prompt_defines_central_assertion():
    p = fc.VERIFICATION_SYSTEM_PROMPT
    assert "central assertion" in p and '"but"' not in p and fc.PROMPT_VERSION.startswith("v3")


# cost controls


class FakeBatches:
    def __init__(self, replies):
        self.replies, self.created = replies, []

    def create(self, requests):
        self.created.append(list(requests))
        return NS(id="batch_1")

    def retrieve(self, _id):
        return NS(processing_status="ended",
                  request_counts=NS(succeeded=len(self.replies), processing=0, errored=0, expired=0))

    def results(self, _id):
        for cid, msg in self.replies.items():
            if msg is None:
                yield NS(custom_id=cid, result=NS(type="expired"))
            else:
                yield NS(custom_id=cid, result=NS(type="succeeded", message=msg))


def _dataset(tmp_path, n=3):
    import json as _j
    (tmp_path / "d.json").write_text(_j.dumps([{"claim": f"c{i}", "label": "Refuted"} for i in range(n)]))


def test_batch_run_uses_batch_replies_and_halves_cost(tmp_path, monkeypatch, fake):
    import json as _j
    _dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    ok = resp(text(verdict(verdict="likely_false")))
    ok.usage = NS(input_tokens=40000, output_tokens=1000, server_tool_use=NS(web_search_requests=2))
    no_block = resp(text("forgot the block"))
    no_block.usage = NS(input_tokens=40000, output_tokens=1000, server_tool_use=NS(web_search_requests=2))
    client = fake([resp(text(verdict(verdict="likely_false")))])  # the one follow-up
    client.batches = FakeBatches({"claim-0": ok, "claim-1": no_block, "claim-2": None})
    ev.main(["run", "--data", "d.json", "--batch", "--yes", "--workers", "1"])
    params = client.batches.created[0][0]["params"]
    assert params["tools"][0]["max_uses"] == 3 and "blocked_domains" in params["tools"][0]
    recs = {r["id"]: r for r in map(_j.loads, open("results/averitec_search_blocked.jsonl"))}
    assert recs[0]["correct"] and recs[0]["batch"] and recs[1]["correct"]
    assert "expired" in recs[2]["error"]
    assert len(client.calls) == 1  # only claim 1 needed a direct follow-up
    assert abs(recs[0]["cost_usd"] - (0.5 * (40000 * 2 + 1000 * 10) / 1e6 + 0.02)) < 1e-6
    assert not os.path.exists("results/averitec_search_blocked.batch.json")


def test_batch_resume_does_not_resubmit(tmp_path, monkeypatch, fake):
    import json as _j
    _dataset(tmp_path, 1)
    monkeypatch.chdir(tmp_path)
    os.makedirs("results")
    _j.dump({"batch_id": "batch_1", "ids": [0]}, open("results/averitec_search_blocked.batch.json", "w"))
    client = fake([])
    client.batches = FakeBatches({"claim-0": resp(text(verdict(verdict="likely_false")))})
    ev.main(["run", "--data", "d.json", "--batch", "--workers", "1"])  # no --yes needed when resuming
    assert client.batches.created == []


def test_budget_stops_run(tmp_path, monkeypatch, fake):
    _dataset(tmp_path, 5)
    monkeypatch.chdir(tmp_path)
    r = resp(text(verdict(verdict="likely_false")))
    r.usage = NS(input_tokens=500000, output_tokens=0, server_tool_use=NS(web_search_requests=0))
    fake([r] * 5)
    with pytest.raises(SystemExit) as e:
        ev.main(["run", "--data", "d.json", "--budget", "1.5", "--yes", "--workers", "1"])
    assert "budget" in str(e.value)
    assert len(open("results/averitec_search_blocked.jsonl").read().splitlines()) == 2


def test_dry_run_and_decline_send_nothing(tmp_path, monkeypatch, fake):
    _dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    c = fake([])
    ev.main(["run", "--data", "d.json", "--dry-run"])
    ev.main(["run", "--data", "d.json"])  # no terminal and no --yes: refuses
    assert c.calls == [] and not os.path.exists("results/averitec_search_blocked.jsonl")


def test_estimate_matches_prices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    full, _ = ev.estimate_cost(100, "search_blocked", 5, batch=False)
    half, _ = ev.estimate_cost(100, "search_blocked", 5, batch=True)
    assert abs(full - 100 * ((48500 * 2 + 1700 * 10) / 1e6 + 0.026)) < 0.01
    assert half < full * 0.62
    capped, basis = ev.estimate_cost(100, "search_blocked", 2, batch=True)
    assert capped < half and "at most 2" in basis


# stage 2 and 3 evaluation

import eval_common as ec  # noqa: E402
import eval_images as ei  # noqa: E402
import eval_c2pa as e3  # noqa: E402


def test_reverse_search_withholds_blocked_sites(monkeypatch):
    class R:
        def raise_for_status(self): pass
        def json(self):
            return {"visual_matches": [{"title": "a", "source": "Snopes", "link": "https://www.snopes.com/fact-check/x"},
                                       {"title": "b", "source": "BBC", "link": "https://www.bbc.com/news/y"}]}
    monkeypatch.setattr(fc, "SERPAPI_KEY", "k")
    monkeypatch.setattr(fc.requests, "get", lambda *a, **k: R())
    out = fc.reverse_image_search("https://img", blocked_domains=["snopes.com"])
    assert [m["link"] for m in out["matches"]] == ["https://www.bbc.com/news/y"] and "withheld" in out["note"]


def test_macro_f1_penalises_predictions_in_absent_class():
    recs = [{"gold": "a", "pred": "b", "correct": False}, {"gold": "a", "pred": "a", "correct": True}]
    assert ec.macro_f1(recs, ["a"]) < 0.5


def _images_setup(tmp_path, monkeypatch, n=6):
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    rows = []
    for i in range(n):
        p = tmp_path / f"src{i}.png"
        Image.new("RGB", (1600, 900), (i * 30, 90, 120)).save(p)
        rows.append({"id": f"img{i}", "label": ei.GOLD[i % 3], "caption": f"caption {i}",
                     "image_url": f"https://example.com/{i}.jpg", "image_path": str(p), "source": "test",
                     "fact_check_url": "https://www.snopes.com/fact-check/leak" if i == 0 else ""})
    ec.write_csv("data/images.csv", rows, ei.FIELDS)
    ei.main(["prepare"])


def test_images_prepare_and_classifier_only(tmp_path, monkeypatch):
    from PIL import Image
    _images_setup(tmp_path, monkeypatch)
    prepared = ec.read_csv(ei.PREPARED)
    assert all(r["status"] == "ok" for r in prepared)
    assert max(Image.open(prepared[0]["small_path"]).size) == 1000
    monkeypatch.setattr(fc, "classify_manipulation", lambda p: {"predictions": [{"label": "Fake", "score": 0.9}, {"label": "Real", "score": 0.1}]})
    ei.main(["run", "--condition", "classifier_only"])
    recs = [json.loads(l) for l in open("results/images_classifier_only.jsonl")]
    assert all(r["pred"] == "content_manipulation" and r["cost_usd"] == 0 for r in recs)


def test_images_full_and_no_image(tmp_path, monkeypatch, fake):
    _images_setup(tmp_path, monkeypatch, n=3)
    monkeypatch.setattr(fc, "classify_manipulation", lambda p: {"predictions": [{"label": "Real", "score": 0.8}]})
    monkeypatch.setattr(fc, "reverse_image_search",
                        lambda url, blocked_domains=None, num_results=8: {"matches": [{"link": "https://www.snopes.com/fact-check/leak"}]})
    script = []
    for _ in range(3):
        script += [resp(tool_use("reverse_image_search", "a"), tool_use("classify_manipulation", "b"), stop="tool_use"),
                   resp(text(verdict(category="miscontextualised", verdict="likely_false")))]
    c = fake(script)
    ei.main(["run", "--condition", "full", "--yes", "--workers", "1"])
    first = c.calls[0]["messages"][0]["content"]
    assert first[0]["type"] == "image"
    recs = {r["id"]: r for r in map(json.loads, open("results/images_full.jsonl"))}
    assert recs["img0"]["leak_exact"] and recs["img1"]["correct"] and recs["img0"]["trace"]["serpapi_searches"] == 1
    c2 = fake([resp(tool_use("classify_manipulation", "b"), stop="tool_use"), resp(text(verdict(category="unverifiable")))] * 3)
    ei.main(["run", "--condition", "no_image", "--yes", "--workers", "1"])
    assert all(b["type"] != "image" for b in c2.calls[0]["messages"][0]["content"])
    assert "not shown to you" in c2.calls[0]["system"]
    ei.main(["score"])
    assert os.path.exists("results/images_report/summary.md")


def test_c2pa_set_check_rule_and_model(tmp_path, monkeypatch, fake):
    pytest.importorskip("c2pa")
    pytest.importorskip("cryptography")
    import shutil
    shutil.copytree(os.path.join(os.path.dirname(__file__), "c2pa_test_certs"), tmp_path / "c2pa_test_certs")
    monkeypatch.chdir(tmp_path)
    e3.main(["make-set", "--per-condition", "2"])
    rows = ec.read_csv(e3.MANIFEST)
    assert len(rows) == 10
    e3.main(["check"])
    chk = ec.read_csv("results/c2pa_check.csv")
    assert all(r["matches_expected"] == "yes" for r in chk)
    e3.main(["run", "--condition", "rule"])
    rule = {r["id"]: r for r in map(json.loads, open("results/c2pa_rule.jsonl"))}
    for r in rows:
        want = {"signed_capture": "authentic_truthful", "ai_generated": "content_manipulation",
                "tampered": "content_manipulation", "stripped": "unverifiable", "unsigned": "unverifiable"}[r["condition"]]
        assert rule[r["id"]]["pred"] == want
    c = fake([resp(tool_use("check_c2pa_manifest"), stop="tool_use"), resp(text(verdict(category="unverifiable")))] * 10)
    e3.main(["run", "--condition", "model", "--yes", "--workers", "1"])
    assert all("asset_" not in call["messages"][0]["content"] for call in c.calls)  # real names never shown
    e3.main(["score"])
    assert "Accuracy by type of file" in open("results/c2pa_report/summary.md").read()


import build_image_set as bis  # noqa: E402


def _commons_page(i, year, theme="flood"):
    return {"title": f"File:{theme}_{i}.jpg", "imageinfo": [{
        "mime": "image/jpeg", "width": 2000, "url": f"https://upload.wikimedia.org/{theme}_{i}.jpg",
        "thumburl": f"https://upload.wikimedia.org/thumb/{theme}_{i}.jpg",
        "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{theme}_{i}.jpg",
        "extmetadata": {"ImageDescription": {"value": f"<p>Flood waters over the main road in town number {i}</p>"},
                        "DateTimeOriginal": {"value": f"{year}-03-0{1 + i % 8} 10:00"},
                        "LicenseShortName": {"value": "CC BY-SA 4.0"}, "Artist": {"value": "<a>Someone</a>"}}}]}


def test_commons_builder_true_and_swapped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pages = [_commons_page(i, 2010 + i) for i in range(12)] + [
        {"title": "File:x.svg", "imageinfo": [{"mime": "image/svg+xml", "width": 10}]}]
    monkeypatch.setattr(bis, "_get", lambda params: {"query": {"pages": pages}})
    bis.main(["commons", "--themes", "flood", "--per-label", "2", "--oversample", "2", "--pages", "1"])
    rows = ec.read_csv(bis.COMMONS_OUT)
    true = [r for r in rows if r["group"] == "original_caption"]
    swap = [r for r in rows if r["group"] == "caption_swap"]
    assert len(swap) == 2 and len(true) == 4
    for r in swap:  # the caption belongs to a photo from a different year
        img_year = int(re.search(r"\((\d{4});", r["notes"]).group(1))
        assert str(img_year) not in r["caption"]
    used = [r["image_url"] for r in rows]
    assert len(used) == len(set(used)) and all("<" not in r["caption"] for r in rows)


def test_combine_interleaves_sources_and_prepare_limits(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    a = [{"id": f"a{i}", "label": "miscontextualised", "image_url": f"https://x/a{i}.jpg", "source": "AMMeBa", "group": "self_captioned"} for i in range(4)]
    c = [{"id": f"c{i}", "label": "miscontextualised", "image_url": f"https://x/c{i}.jpg", "source": "Wikimedia Commons", "group": "caption_swap"} for i in range(4)]
    ec.write_csv(bis.AMMEBA_OUT, a, ei.FIELDS + ["group"])
    ec.write_csv(bis.COMMONS_OUT, c, ei.FIELDS + ["group"])
    bis.main(["combine"])
    ids = [r["id"] for r in ec.read_csv(bis.COMBINED)]
    assert ids[:4] == ["a0", "c0", "a1", "c1"]

    def fake_download(url, dest):
        if url.endswith("a0.jpg"):
            raise OSError("404")
        Image.new("RGB", (50, 50)).save(dest, "PNG")
        return dest
    monkeypatch.setattr(ei, "_download", fake_download)
    ei.main(["prepare", "--per-label", "3"])
    st = {r["id"]: r["status"] for r in ec.read_csv(ei.PREPARED)}
    assert st["a0"].startswith("failed") and [k for k, v in st.items() if v == "ok"] == ["c0", "a1", "c1"]
    assert all(v == "spare" for k, v in st.items() if k in ("a2", "c2", "a3", "c3"))


# stage 2 prompt img-v3/v4

def test_image_model_sees_calibrated_score_and_date(fake, jpg, monkeypatch):
    import datetime as dt
    monkeypatch.setattr(fc, "classify_manipulation",
                        lambda p: {"predictions": [{"label": "Deepfake", "score": 0.66}, {"label": "Realism", "score": 0.34}]})
    c = fake([resp(tool_use("classify_manipulation", "b"), stop="tool_use"),
              resp(text(verdict(category="authentic_truthful")))])
    trace = fc.new_trace()
    fc.verify_image(jpg, caption="Flood in Lismore, 2022", trace=trace)
    first = c.calls[0]["messages"][0]["content"]
    assert dt.date.today().isoformat() in first[-1]["text"]
    tool_result = c.calls[1]["messages"][-1]["content"][0]["content"]
    assert "deepfake_score" in tool_result and "0.66" in tool_result and "calibration_note" in tool_result
    assert "Realism" not in tool_result                       # the model no longer sees the top label
    assert trace["tool_results"][0]["predictions"][0]["label"] == "Deepfake"   # the log keeps everything
    assert fc.IMAGE_PROMPT_VERSION == "img-v4"


def test_build_image_set_excludes_earlier_set(tmp_path):
    import eval_common as ec
    import build_image_set as b
    p = tmp_path / "old.csv"
    ec.write_csv(str(p), [
        {"id": "ammeba_x", "image_url": "https://x/1.jpg", "notes": "image_id=abc123"},
        {"id": "commons_swap_001", "image_url": "https://up/2.jpg",
         "notes": "image https://commons.wikimedia.org/wiki/File:A_(b).jpg (2019; CC0; X); "
                  "caption from https://commons.wikimedia.org/wiki/File:C.jpg (2016)"},
    ], ["id", "image_url", "notes"])
    u = b._used([str(p)])
    assert u["ammeba"] == {"abc123"}
    assert u["pages"] == {"https://commons.wikimedia.org/wiki/File:A_(b).jpg",
                          "https://commons.wikimedia.org/wiki/File:C.jpg"}
    assert b._paths("test")[2].endswith("images_test.csv")


def test_image_eval_stops_when_reverse_search_fails(monkeypatch, jpg):
    import eval_images as ei
    import eval_averitec as ev

    def fake_verify(path, *, trace=None, **kw):
        trace.setdefault("tool_results", []).append(
            {"tool": "reverse_image_search", "error": "reverse image search request failed: 429 Client Error: Too Many Requests"})
        return fc.ImageVerdict(image_ref=path, category="authentic_truthful", verdict="likely_true",
                               confidence="medium", justification="x")
    monkeypatch.setattr(fc, "verify_image", fake_verify)
    args = NS(label="full", condition="full", no_block=False)
    rec = ei.evaluate({"id": "a", "label": "authentic_truthful", "image_path": jpg, "image_url": "https://x/a.jpg"}, args)
    assert rec["error"].startswith("RuntimeError: SerpApi unavailable") and rec["pred"] is None
    assert ev.FATAL_ERROR_RE.search(rec["error"])


# recent (post-cutoff) claims

def test_recent_rating_map():
    import build_recent_claims as b
    cases = {"False": "Refuted", "Pants on Fire!": "Refuted", "Mostly False": "Refuted",
             "Half True": "Conflicting Evidence/Cherrypicking", "Missing context": "Conflicting Evidence/Cherrypicking",
             "Misleading": "Conflicting Evidence/Cherrypicking", "Mostly True": "Supported", "True": "Supported",
             "Correct": "Supported", "Unproven": "Not Enough Evidence", "No evidence": "Not Enough Evidence",
             "Satire": None, "Labeled Satire": None, "": None, "Explainer": None}
    for rating, want in cases.items():
        assert b.map_rating(rating) == want, rating


def test_recent_fetch_filters_and_writes_averitec_format(tmp_path, monkeypatch):
    import datetime as dt
    import build_recent_claims as b
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_FACTCHECK_API_KEY", "x")
    rev = lambda url, rating, day="2026-05-02T00:00:00Z": {"url": url, "textualRating": rating, "reviewDate": day,
                                                           "languageCode": "en", "publisher": {"site": "politifact.com"}}
    reply = {"claims": [
        {"text": "The new budget cuts school funding by 40 per cent", "claimant": "A Senator",
         "claimDate": "2026-04-28T00:00:00Z", "claimReview": [rev("https://www.politifact.com/factchecks/2026/may/02/a/", "False")]},
        {"text": "Old claim from before the cutoff about tax rates", "claimDate": "2025-12-01T00:00:00Z",
         "claimReview": [rev("https://www.politifact.com/factchecks/2026/may/03/b/", "False")]},
        {"text": "Video shows a flood in the capital last week", "claimReview": [rev("https://www.politifact.com/factchecks/2026/may/04/c/", "False")]},
        {"text": "Claim reviewed on an unblocked site about something", "claimReview": [rev("https://example.com/check/1", "False")]},
        {"text": "A claim with a rating nobody can map cleanly", "claimReview": [rev("https://www.politifact.com/factchecks/2026/may/05/d/", "Explainer")]},
    ]}
    monkeypatch.setattr(b, "_get", lambda params: reply)
    b.main(["fetch", "--publishers", "politifact.com", "--since", "2026-04-01", "--min-claims", "1"])
    data = json.load(open("data/recent_claims.json"))
    assert len(data) == 1
    item = data[0]
    assert item["label"] == "Refuted" and item["claim_date"] == "28-04-2026" and item["speaker"] == "A Senator"
    assert item["fact_checking_article"].startswith("https://www.politifact.com/")
    import eval_averitec as ev
    assert ev.load("data/recent_claims.json")[0]["_id"] == 0 and item["label"] in ev.LABEL_MAP


def test_recent_sample_keeps_minority_labels():
    import build_recent_claims as b
    pool = ([{"label": "Refuted", "review_date": f"2026-05-{i % 28 + 1:02d}", "fact_checking_article": f"r{i}"} for i in range(300)]
            + [{"label": "Supported", "review_date": "2026-06-01", "fact_checking_article": f"s{i}"} for i in range(20)]
            + [{"label": "Not Enough Evidence", "review_date": "2026-06-01", "fact_checking_article": f"n{i}"} for i in range(10)]
            + [{"label": "Conflicting Evidence/Cherrypicking", "review_date": "2026-06-01", "fact_checking_article": f"c{i}"} for i in range(80)])
    out = b.choose(pool, 150, 42)
    from collections import Counter
    c = Counter(x["label"] for x in out)
    assert len(out) == 150 and c["Supported"] == 20 and c["Not Enough Evidence"] == 10
    assert c["Conflicting Evidence/Cherrypicking"] == 45 and c["Refuted"] == 75
    assert b.choose(pool, 150, 42) == out
    assert b.map_rating("this is a hoax. there's no evidence") == "Refuted"
    assert b.map_rating("False. There is no evidence that") == "Refuted"

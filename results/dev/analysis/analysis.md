# Further analysis of AVeriTeC results

158 scored claims from search_blocked, prompt version(s) v2-misleading-attribution, model claude-sonnet-5. Error records are excluded.

## 1. Telling true from false

| Claims AVeriTeC labels Supported or Refuted | Value |
|---|---|
| Claims | 133 |
| Pipeline committed to true or false | 104 (78%) |
| Accuracy when it committed | 94% (98/104) |
| Labelled misleading or unverifiable instead | 29 |
| Accuracy on all of these claims | 74% (98/133) |

## 2. Accuracy by claim characteristics

Groups can overlap (a claim can have several types or strategies). Groups of fewer than 5 claims are listed but too small to interpret.

**AVeriTeC claim type**

| Group | Claims | Accuracy |
|---|---|---|
| Event/Property Claim | 97 | 69% (67/97) |
| Numerical Claim | 40 | 55% (22/40) |
| Quote Verification | 28 | 54% (15/28) |
| Causal Claim | 19 | 68% (13/19) |
| Position Statement | 9 | 67% (6/9) |

**AVeriTeC fact-checking strategy**

| Group | Claims | Accuracy |
|---|---|---|
| Written Evidence | 145 | 64% (93/145) |
| Consultation | 40 | 68% (27/40) |
| Numerical Comparison | 33 | 61% (20/33) |
| Fact-checker Reference | 13 | 100% (13/13) |
| Satirical Source Identification | 3 | 100% (3/3) |

**Country (ISO code)**

| Group | Claims | Accuracy |
|---|---|---|
| US | 74 | 65% (48/74) |
| IN | 22 | 68% (15/22) |
| NG | 18 | 50% (9/18) |
| (none) | 13 | 85% (11/13) |
| GB | 8 | 50% (4/8) |
| NZ | 4 | 50% (2/4) |
| FR | 3 | 100% (3/3) |
| CN | 3 | 33% (1/3) |
| AU | 3 | 67% (2/3) |
| LK | 2 | 100% (2/2) |
| CA | 2 | 100% (2/2) |
| PK | 1 | 100% (1/1) |
| AT | 1 | 100% (1/1) |
| DE | 1 | 100% (1/1) |
| PH | 1 | 0% (0/1) |
| RU | 1 | 100% (1/1) |
| TR | 1 | 100% (1/1) |

**Where the claim appeared**

| Group | Claims | Accuracy |
|---|---|---|
| Facebook | 51 | 65% (33/51) |
| (none) | 23 | 74% (17/23) |
| Twitter | 16 | 88% (14/16) |
| Instagram | 6 | 50% (3/6) |
| Channels Television | 4 | 75% (3/4) |
| US televised debate on Reuters. | 3 | 0% (0/3) |
| Independence Anniversary Speech | 3 | 67% (2/3) |
| Youtube | 2 | 50% (1/2) |
| twitter | 2 | 50% (1/2) |
| Channels TV | 2 | 0% (0/2) |
| CNN | 2 | 100% (2/2) |
| The Federalist | 1 | 0% (0/1) |
| Twitter/Latest laws | 1 | 100% (1/1) |
| The Sun  | 1 | 100% (1/1) |
| Speech, Republican National Convention | 1 | 100% (1/1) |
| Final Presedential Debate | 1 | 0% (0/1) |
| New York Post | 1 | 100% (1/1) |
| Natural News | 1 | 0% (0/1) |
| Zee news, DNA, Times now and First post | 1 | 0% (0/1) |
| media outlets | 1 | 0% (0/1) |
| Facebook User posts | 1 | 0% (0/1) |
| Journal of the American Medical Association (JAMA) | 1 | 100% (1/1) |
| Washington Examiner | 1 | 0% (0/1) |
| WhatsApp viral message and social media | 1 | 100% (1/1) |
| Authority Newspaper | 1 | 0% (0/1) |
| Social Media | 1 | 100% (1/1) |
| KCRG | 1 | 100% (1/1) |
| US televised debate on Reuters | 1 | 0% (0/1) |
| Radio Canada | 1 | 100% (1/1) |
| Russian Foreign Ministry | 1 | 100% (1/1) |
| Twitter  | 1 | 0% (0/1) |
| Fox News | 1 | 100% (1/1) |
| ChannelsTV | 1 | 100% (1/1) |
| Slate | 1 | 0% (0/1) |
| Newshub Leaders Debate  | 1 | 100% (1/1) |
| FACEBOOK | 1 | 100% (1/1) |
| NBC NEWS | 1 | 100% (1/1) |
| Public forum | 1 | 100% (1/1) |
| Social Media  | 1 | 100% (1/1) |
| Facebook and whatapp | 1 | 100% (1/1) |
| Republic World | 1 | 0% (0/1) |
| Facebook Users Posts | 1 | 100% (1/1) |
| U.S. State Department | 1 | 100% (1/1) |
| instagram | 1 | 100% (1/1) |
| facebook User Post. | 1 | 100% (1/1) |
| CBC News | 1 | 0% (0/1) |
| Press Release | 1 | 0% (0/1) |
| transmitted via Facebook and Zoom, University of Ibadan, Nigeria's UI Community Forum's Voice of Next Vice-Chancellor 2020 | 1 | 100% (1/1) |
| Twitter - Times Now | 1 | 100% (1/1) |
| Toronto today | 1 | 100% (1/1) |
| Tv election debate on 'Channels Television' | 1 | 0% (0/1) |
| Rev | 1 | 100% (1/1) |
| Sudarshan News | 1 | 100% (1/1) |
| tvanouvelles | 1 | 100% (1/1) |
| Times Now | 1 | 0% (0/1) |

**Claim length**

| Group | Claims | Accuracy |
|---|---|---|
| under 15 words | 72 | 61% (44/72) |
| 15 to 24 words | 59 | 69% (41/59) |
| 25 words or more | 27 | 70% (19/27) |

**Gold label**

| Group | Claims | Accuracy |
|---|---|---|
| Refuted | 93 | 73% (68/93) |
| Supported | 40 | 75% (30/40) |
| Not Enough Evidence | 13 | 8% (1/13) |
| Conflicting Evidence/Cherrypicking | 12 | 42% (5/12) |

## 3. Stated confidence against accuracy

| Confidence | Claims | Accuracy |
|---|---|---|
| high | 117 | 74% (87/117) |
| medium | 39 | 44% (17/39) |
| low | 2 | 0% (0/2) |

## 4. Search effort, cost and accuracy

| Searches used | Claims | Mean input tokens | Mean cost (USD) | Accuracy |
|---|---|---|---|---|
| 1 | 11 | 17,127 | 0.054 | 82% (9/11) |
| 2 | 94 | 29,789 | 0.094 | 68% (64/94) |
| 3 | 21 | 62,900 | 0.176 | 57% (12/21) |
| 4 | 15 | 88,537 | 0.241 | 53% (8/15) |
| 5 | 17 | 119,248 | 0.313 | 65% (11/17) |

Claims using 4 or more searches were 20% of claims but 44% of input tokens. Input tokens rise faster than the number of searches because each further search re-reads everything gathered so far.

Estimated total cost of these 158 claims: US$22.05.

## 5. Sources

| Type of site retrieved | Share of retrieved pages |
|---|---|
| News and other | 73% |
| Government and intergovernmental | 13% |
| Social media and video | 8% |
| Academic | 4% |
| Wikipedia | 3% |

2,580 pages retrieved in total, 16.3 per claim. Claims cited 3.3 sources on average. 514 of 518 cited URLs (99.2%) were among the pages the search actually returned; the rest may be invented or recalled from training.

## 6. Predicted against gold label distribution

| Label | Predicted | Gold |
|---|---|---|
| likely_true | 39 | 40 |
| likely_false | 75 | 93 |
| misleading | 41 | 12 |
| unverifiable | 3 | 13 |


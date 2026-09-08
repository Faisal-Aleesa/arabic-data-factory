# Task 2 reconstruction output — verification intake findings

**Date:** 2026-09-08 · **Commit reviewed:** `0e44f75` (3 commits, Mohammed AlZiyad)
**Adapter:** `src/verification/adapt_task2.py` (read-only) · **Corpus:** classical only

This is the canonical reference for what the verification layer found on Task 2's first
delivery. It is written to be sent to Task 2 as-is.

**Every finding below carries a confidence marker.** Two of them (§2.1, §2.3) were
downgraded after the fact axis was found unreliable on this content — see §4, and see
§2.0 for which findings depend on it and which do not.

**Read the three sections in order.** Section 1 is a defect that invalidates DPO
measurement until fixed. Section 2 is a short list of things that are actually wrong with
the data. Section 3 is a longer list of things that look wrong but are mostly a
convention mismatch between our two stages — and section 4 explains why most of the
alarming fact-checking numbers should not be trusted yet.

Nothing here was run against the dialect corpus, because **Task 2's output contains no
dialect records at all** (measured: 394/394 classical chunks, 0/337 dialect).

---

## 0. What arrived

| file | records | note |
|---|---:|---|
| `data/generated/sft/accepted.jsonl` | 4,645 | |
| `data/generated/sft/candidates.jsonl` | 4,645 | **byte-identical** to `accepted` |
| `data/generated/sft/candidates copy.jsonl` | 4,645 | **byte-identical**; stray file |
| `data/generated/dpo/candidates.jsonl` | 3,000 | |

All JSON parses; 0 malformed lines. Nothing existing in the repo was deleted or modified
by the merge (0 D, 0 M, 16 A).

`accepted.jsonl` is not a filtered subset — it is a copy. Every record carries
`validation_status: "Unverified"`. **Either no acceptance filtering ran, or the file is
misnamed.** Please confirm which, and delete `candidates copy.jsonl`.

Coverage: SFT touches **394/394 classical chunks (100%)**, ~11.8 records each. DPO touches
**256/394 (65%)**, 1 pair per source sample.

---

## 1. URGENT — the `Thinking:` scaffold makes DPO unmeasurable

**Measured: `chosen` begins with a literal `Thinking:\n…\n\nAnswer:\n` scaffold in
2,485 of 3,000 pairs. `rejected` carries it in ZERO of 3,000.**

This is a formatting difference present in 83% of pairs and **perfectly correlated with
the label**. Consequences, in order of severity:

1. **A preference model can learn the scaffold instead of the task.** "Prefer the response
   that starts with `Thinking:`" scores 83% without reading a word of Arabic. Any DPO
   training on this data risks learning exactly that, and it would look like success.
2. **Every automated check is confounded.** Format conformance, length ratio and lexical
   distinctness in `check_dpo.py` would all be measuring the scaffold rather than the
   declared weakness.
3. It is asymmetric with the SFT side, where `response` is the answer with no scaffold.

**This is the one item we would ask you to fix before regenerating anything else.** The
fix belongs in the generator: either emit the scaffold on both sides or on neither.

Our adapter strips it by default (`--chosen-part answer`) so the numbers below are not
contaminated. That is a workaround for measurement, **not a fix** — the shipped files
still contain it.

---

## 2. Defects — with confidence markers

### 2.0 Which findings depend on the fact axis, and which do not

Section 4 shows `check_facts.py` mis-scores conversational paraphrase. `check_dpo`'s
`direction` field consumes those verdicts directly, so anything derived from `direction`
inherits the defect. **Measured: with similarity off, 100% of `tie` outcomes are pure
verdict-identity ties — 0 pairs tied despite differing verdicts.** The fact axis is not
one input among several here; it is the only one.

| finding | depends on `check_facts`? | confidence |
|---|---|---|
| §1 scaffold asymmetry | no — plain string inspection | **HIGH** |
| §2.2 degenerate pairs | no — lexical distinctness only | **HIGH** |
| §2.4 one rejection type | no — metadata count | **HIGH** |
| §3.1 `format_type` | no — `check_format`, independent module | **HIGH** |
| §0 duplicate files | no — byte comparison | **HIGH** |
| **§2.1 inverted pairs** | **yes** | **LOW — see below** |
| **§2.3 indistinguishable pairs** | **yes** | **LOW — see below** |


### 2.1 139 "inverted" pairs — CONFIDENCE LOW, mostly an artifact

`direction = rejected_better_verdict` on 139 of 3,000. An earlier draft of this document
reported these as inverted or mislabelled pairs. **That was wrong, and the correction
matters more than the original finding.**

Provenance of all 139:

| chosen verdict | rejected verdict | count |
|---|---|---:|
| REVIEW | NO_FACT_COVERAGE | 73 |
| FAIL_CONTRADICTED | NO_FACT_COVERAGE | 47 |
| FAIL_CONTRADICTED | REVIEW | 19 |

**In 120 of 139 (86%), `rejected` "wins" because the fact axis found nothing in it to
check.** It did not verify better; it escaped scrutiny. Meanwhile `chosen` was examined
and flagged — and §4 measures those flags as ~87% morphological false positives.

The mechanism is a direct inversion driver: a `chosen` response that mentions a lexical
root trips the morphology bug and is penalised; a `rejected` response that happens not to
mention one is silent and therefore "better". Corroborating measurement — **only-rejected-
silent = 120, exactly matching the inversions; only-chosen-silent = 45, which produces the
opposite direction and no inversion.**

Length is NOT the explanation: within these 139, mean words are chosen 18.7 vs rejected
18.2. The two sides are the same size; they differ in whether they happened to contain a
root-shaped token.

**How many are genuinely inverted is currently unknown.** The 19 `FAIL_CONTRADICTED` vs
`REVIEW` pairs are the least contaminated subset and the place to look first — but they
rest on the same flags. Re-measure after the morphology fix before sending any number
here to Task 2 as a defect count.

### 2.2 2 degenerate pairs

Lexical distinctness ≥ 0.995 — `chosen` and `rejected` are essentially the same text, so
the pair carries no contrastive signal.

### 2.3 2,801 "indistinguishable" pairs — CONFIDENCE LOW, largely tied by silence

All 3,000 pairs declare `partial_factual_errors`, which `check_dpo.DETECTABILITY` rates
**`strong`**. The checks separated only 58 pairs in favour of `chosen`; 2,803 came back
`tie`. An earlier draft read that as "the injected errors are too subtle or were not
injected". **That reading is not supportable.**

How every tie was actually reached:

| chosen verdict | rejected verdict | count |
|---|---|---:|
| NO_FACT_COVERAGE | NO_FACT_COVERAGE | **2,092 (75%)** |
| FAIL_CONTRADICTED | FAIL_CONTRADICTED | 367 |
| REVIEW | REVIEW | 344 |
| *(differing verdicts)* | | **0** |

**75% of ties are the fact axis saying nothing about EITHER side.** That is a tie by
absence of measurement, not a finding of equivalence. A further 367 are both sides
tripping the same morphology bug.

And the pairs are demonstrably not identical text. Lexical distinctness across the 2,803
ties: median **0.370**, and **2,286 (82%) below 0.60** — substantially different wording.
Only 2 sit at the degenerate floor. So the pairs differ; the checker cannot see how.

**This number says more about our checker than about Task 2's data.** It should not be
sent as a defect count until the fact axis works on conversational text.

### 2.4 Only 1 of 9 charter rejection types was produced

| type | count |
|---|---:|
| `partial_factual_errors` | 3,000 |
| the other 8 | **0** |

`wrong_register` and `weak_organization` in particular have **zero** deterministic
coverage anywhere in the project — the LLM judge is their only check, and it cannot be
exercised at all without pairs of those types.

---

## 3. Structural mismatches — large numbers, mostly NOT bad data

### 3.1 `format_type` — 3,484 rejections from a labelling convention

`format_conforms` is **false on 3,606 of 4,645 records (78%)**, which drives 3,484 of the
4,143 rejections. The cause is not bad responses.

Every record declares `format_type: dictionary_entry`, copied from the source chunk. The
responses are conversational answers, so they do not have dictionary-entry shape and the
check correctly says so.

`src/deployment/ACCEPTANCE_CONTRACT.md` predicted this in writing, before the data
existed:

> If reconstruction copies `format_type` from the chunk, then a summarisation instruction
> over a `dictionary_entry` chunk produces flowing prose labelled `dictionary_entry`,
> every such record mismatches, and the gate rejects work that is perfectly good.

**The project policy is that `format_type` describes the shape the RESPONSE is intended to
have — not an inherited copy of the chunk's format.** That policy is what makes the
hard-reject tier defensible. Two ways forward, and this is Task 2's call:

- set `format_type` to the response's actual intended shape (`prose` for most of these), or
- tell us it must mirror the chunk, in which case the contract's `dictionary_entry`
  hard-reject tier has to be revisited before this ships.

**Do not read 4,143 rejections as 4,143 bad records.**

### 3.2 `NO_CHECKABLE_CLAIMS` — 3,333 records (72%)

`check_facts.py` found no checkable assertions in 72% of responses. This is a property of
the checker meeting a kind of text it was not built for — see section 4 — not evidence
that the responses are empty or wrong.

---

## 4. CAVEAT — `check_facts.py` has NOT been validated on conversational paraphrase

**Read this before quoting any fact-checking number from this document.**

`extract_facts.py` and `check_facts.py` were built and measured against **dictionary-entry
prose**: headword, gloss, cited authority. They have never been validated against
conversational Saudi-dialect paraphrase, which is what Task 2 generates. The intake run is
the first time they have met this kind of text, and the evidence says they are
mis-scoring it.

**The measurement that shows it:**

| flagged verdict | flagged strings that appear VERBATIM in their own source chunk |
|---|---|
| `CONTRADICTED` | **625 / 715 (87%)** |
| `UNSUPPORTED` | **626 / 801 (78%)** |

A response string that is literally present in the source chunk is not, on its face, a
contradiction of it. **100% of these flags are `entry_root`.**

The mechanism looks like morphology. Example from the sample below: the fact table holds
the root `أثف`; the response wrote `أثفية`. `check_facts._relation()` sees added characters
that no token run aligns, classifies it as `corruption`, and returns `CONTRADICTED`. But
`أثفية` is an ordinary derived form of `أثف` — precisely what a lexicon entry documents.
The checker has no morphological analyser and was never asked to be one, because
dictionary-entry text quotes roots verbatim and conversational paraphrase inflects them.

**Therefore:**

- The **591 CONTRADICTED** and **530 UNSUPPORTED** record-level verdicts should be treated
  as an upper bound with a large false-positive component, not as a defect count.
- The residual worth investigating is the ~90 CONTRADICTED whose flagged string does
  **not** appear verbatim in its chunk. That is the population most likely to contain
  genuine invention — but do not assume it is clean either. Cases 2 and 3 of the spot
  check below are both in that residual, and both are still morphological: `مددت الدواة`
  flagged against root `مدد`, and `هودج الأجلح` against root `جلح`. The string is absent
  verbatim only because the response inflected the root and attached it to a phrase.
  **The true false-positive rate is therefore probably higher than 87%, not lower.**
- The **44 SUPPORTED** records are correspondingly a lower bound.
- This is a defect in **our** verification layer's coverage, not in Task 2's output, and
  we own fixing it. Until it is fixed, the fact axis is not a usable quality gate for
  conversational responses.

---

## 5. Recommended manual spot check — 18 cases

**Someone should eyeball these before the aggregate numbers in sections 2–4 are treated as
final.** 9 `CONTRADICTED` and 9 `UNSUPPORTED`, drawn by fixed-seed random sample
(seed 20260908) from the 1,121 flagged records, stratified by verdict — **not** the first
N by index, which would have clustered the sample into one or two source chunks.

Spread: **16 distinct source chunks** across 18 cases.

For each case, decide whether the checker is right. The column that matters most is
whether the flagged string is actually in the chunk — where it is, the flag is probably a
morphology false positive of the kind section 4 describes.

Tally the outcome and record it here; if the false-positive rate is as high as section 4
suggests, the fact axis needs the morphology fix before it is run against Task 2 output
again.

---

### 1. `asas_albalagha_c0366` — UNSUPPORTED

| | |
|---|---|
| line in adapted file | 4330 |
| flagged assertion | `خرقاء ذات نيقة` (`entry_root`) |
| checker reason | no entry_root in chunk matches |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** وش معنى "خرقاء ذات نيقة"؟

**Response:** المثل "خرقاء ذات نيقة" يطلق على الشخص الجاهل اللي يدعي المعرفة. يعني هو ما يفهم شيء بس يتفلسف ويسوي نفسه فاهم ودقيق.

**Verdict to confirm or overturn:** _______________

### 2. `asas_albalagha_c0326` — CONTRADICTED

| | |
|---|---|
| line in adapted file | 3851 |
| flagged assertion | `مددت الدواة` (`entry_root`) |
| checker reason | appended/altered characters on a known fact 'مدد' - the source token was modified, not extended |
| that string appears verbatim in the chunk? | **no** |

**Instruction:** كيف ممكن نميز بين "مدّ" و "أمدّ" لما نتكلم عن الدواة؟

**Response:** في سياق الدواة، "مددت الدواة" و "أمددت الدواة بالمداد" كلها تعني إنك عبيتها بالمداد أو زودتها فيه. يعني ما فيه فرق جوهري بينهم في هالحالة.

**Verdict to confirm or overturn:** _______________

### 3. `asas_albalagha_c0046` — CONTRADICTED

| | |
|---|---|
| line in adapted file | 552 |
| flagged assertion | `هودج الأجلح` (`entry_root`) |
| checker reason | appended/altered characters on a known fact 'جلح' - the source token was modified, not extended |
| that string appears verbatim in the chunk? | **no** |

**Instruction:** وش يعني "هودج أجلح"؟

**Response:** الـ"هودج الأجلح" هو الهودج اللي ما فيه قبة، يعني يكون مكشوف وما له سقف.

**Verdict to confirm or overturn:** _______________

### 4. `asas_albalagha_c0383` — UNSUPPORTED

| | |
|---|---|
| line in adapted file | 4527 |
| flagged assertion | `ميضأة` (`entry_root`) |
| checker reason | no entry_root in chunk matches |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** وش يعني "ميضأة"؟

**Response:** الـ"ميضأة" هي المكان أو الأداة اللي يتوضأ منها الواحد، يعني زي مكان الوضوء أو الإناء اللي يستخدم للوضوء.

**Verdict to confirm or overturn:** _______________

### 5. `asas_albalagha_c0214` — CONTRADICTED

| | |
|---|---|
| line in adapted file | 2528 |
| flagged assertion | `طرفة` (`entry_root`) |
| checker reason | appended/altered characters on a known fact 'طرف' - the source token was modified, not extended |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** ليش يقولون عن المرأة "طرفة"؟

**Response:** يقولون عن المرأة "طرفة" إذا كانت ما تثبت على زوج واحد، يعني تدور على رجال ثانيين وما تكتفي بزوجها. زي الناقة "الطرفة" اللي ما تثبت على مرعى واحد.

**Verdict to confirm or overturn:** _______________

### 6. `asas_albalagha_c0338` — UNSUPPORTED

| | |
|---|---|
| line in adapted file | 3992 |
| flagged assertion | `أمات الشيء طبخاً` (`entry_root`) |
| checker reason | no entry_root in chunk matches |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** متى نقول "أمات الشيء طبخاً"؟

**Response:** نقول "أمات الشيء طبخاً" إذا طبخناه زيادة عن اللزوم لين فقد خصائصه أو استوى تماماً، زي ما يقولون "أميتت الخمر" يعني طبخت.

**Verdict to confirm or overturn:** _______________

### 7. `asas_albalagha_c0128` — UNSUPPORTED

| | |
|---|---|
| line in adapted file | 1513 |
| flagged assertion | `هو على الرضف` (`entry_root`) |
| checker reason | no entry_root in chunk matches |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** كيف ممكن نوصف شخص إنه "على الرضف"؟

**Response:** لما نقول "هو على الرضف" يعني إنه قلقان ومضطرب، أو إنه معصب ومغتاظ مرة، كأنه واقف على جمر أو حجارة حارة.

**Verdict to confirm or overturn:** _______________

### 8. `asas_albalagha_c0146` — CONTRADICTED

| | |
|---|---|
| line in adapted file | 1720 |
| flagged assertion | `زبدة العمر` (`entry_root`) |
| checker reason | appended/altered characters on a known fact 'زبد' - the source token was modified, not extended |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** وش يعني "زبدة العمر" في اللهجة السعودية؟

**Response:** تعبير "زبدة العمر" يعني أفضل وأهم جزء في حياة الواحد، أو خلاصة تجاربه وأيامه الحلوة. يعني الشيء اللي له قيمة كبيرة في حياته.

**Verdict to confirm or overturn:** _______________

### 9. `asas_albalagha_c0098` — CONTRADICTED

| | |
|---|---|
| line in adapted file | 1160 |
| flagged assertion | `دردره` (`entry_root`) |
| checker reason | appended/altered characters on a known fact 'درد' - the source token was modified, not extended |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** وش معنى "دردره" أو "درادره" عند الشيخ؟

**Response:** "دردره" أو "درادره" عند الشيخ معناها الأسنان اللي تحاتت أو تآكلت عنده. يعني هي بقايا الأسنان أو اللثة اللي يستخدمها عشان يمضغ الأشياء زي البسرة.

**Verdict to confirm or overturn:** _______________

### 10. `asas_albalagha_c0130` — UNSUPPORTED

| | |
|---|---|
| line in adapted file | 1539 |
| flagged assertion | `وسع الله قدرك` (`entry_root`) |
| checker reason | no entry_root in chunk matches |
| that string appears verbatim in the chunk? | **no** |

**Instruction:** وش معنى "أرغمه الله قدرك"؟

**Response:** يعني "وسع الله قدرك" أو "أبعد خطوك"، وهو دعاء بتوسيع الرزق أو المكانة أو العمر، يعني يزيد من خيرك ومكانتك.

**Verdict to confirm or overturn:** _______________

### 11. `asas_albalagha_c0323` — UNSUPPORTED

| | |
|---|---|
| line in adapted file | 3819 |
| flagged assertion | `ماتنه في الشعر` (`entry_root`) |
| checker reason | no entry_root in chunk matches |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** وش يعني "ماتنه في الشعر"؟

**Response:** لما تقول "ماتنه في الشعر" يعني عارضه أو باراه في الشعر. كأنهم يتنافسون مين اللي شعره أقوى وأمتن، زي تحدي بين شاعرين.

**Verdict to confirm or overturn:** _______________

### 12. `asas_albalagha_c0082` — UNSUPPORTED

| | |
|---|---|
| line in adapted file | 968 |
| flagged assertion | `عيش خروع` (`entry_root`) |
| checker reason | no entry_root in chunk matches |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** وش معنى "عيش خروع" أو "شباب خروع"؟

**Response:** لما نقول "عيش خروع" أو "شباب خروع" يعني عيشة ناعمة ومرفهة، أو شباب فيه نعومة وراحة.

**Verdict to confirm or overturn:** _______________

### 13. `asas_albalagha_c0132` — CONTRADICTED

| | |
|---|---|
| line in adapted file | 1554 |
| flagged assertion | `بالتشديد` (`entry_root`) |
| checker reason | no entry_root in chunk matches |
| that string appears verbatim in the chunk? | **no** |

**Instruction:** إيش الفرق بين "رفع" و "رفّع"؟

**Response:** الفرق إن "رفع" ممكن تكون بمعنى شال شيء أو علاه، زي "رفع القيد" أو "رفع السعر". أما "رفّع" (بالتشديد) تجي بمعنى التقديم أو التنمية، زي "رفّعته إلى السجفين" يعني قدمته، أو "رفعت الرجل" يعني نميته ونسبته.

**Verdict to confirm or overturn:** _______________

### 14. `asas_albalagha_c0099` — CONTRADICTED

| | |
|---|---|
| line in adapted file | 1162 |
| flagged assertion | `ضلّ الدريص نفقه` (`entry_root`) |
| checker reason | appended/altered characters on a known fact 'دري' - the source token was modified, not extended |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** وش يعني "درص" لما نقول "ضلّ الدريص نفقه"؟

**Response:** لما نقول "ضلّ الدريص نفقه"، يعني إن الشخص أخطأ في حجته أو ضيعها.

**Verdict to confirm or overturn:** _______________

### 15. `asas_albalagha_c0214` — CONTRADICTED

| | |
|---|---|
| line in adapted file | 2527 |
| flagged assertion | `طرفت عينه` (`entry_root`) |
| checker reason | appended/altered characters on a known fact 'طرف' - the source token was modified, not extended |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** إيش الفرق بين "طرفت عينه" و "غض طرفه"؟

**Response:** الفرق إن "طرفت عينه" يعني إن العين انصابت بشيء، زي لما يدخل فيها تراب أو يضربها ثوب. أما "غض طرفه" يعني إنه نزل عينه أو صرف بصره عن شيء معين، وهذا يكون بقصده هو.

**Verdict to confirm or overturn:** _______________

### 16. `asas_albalagha_c0003` — CONTRADICTED

| | |
|---|---|
| line in adapted file | 42 |
| flagged assertion | `أزيز` (`entry_root`) |
| checker reason | appended/altered characters on a known fact 'أزى' - the source token was modified, not extended |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** شنو يعني "أزت البرمة"؟

**Response:** يعني البرمة طلعت صوت نشيش أو غليان، زي صوت القدر لما يكون على النار ويغلي. ويسمون هذا الصوت "أزيز".

**Verdict to confirm or overturn:** _______________

### 17. `asas_albalagha_c0152` — UNSUPPORTED

| | |
|---|---|
| line in adapted file | 1801 |
| flagged assertion | `سهم زاهق` (`entry_root`) |
| checker reason | no entry_root in chunk matches |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** إذا قالوا "سهم زاهق"، إيش قصدهم؟

**Response:** "سهم زاهق" يعني سهم تجاوز الهدف اللي كان موجه له ووقع وراه.

**Verdict to confirm or overturn:** _______________

### 18. `asas_albalagha_c0146` — UNSUPPORTED

| | |
|---|---|
| line in adapted file | 1723 |
| flagged assertion | `حرب زبون` (`entry_root`) |
| checker reason | no entry_root in chunk matches |
| that string appears verbatim in the chunk? | **YES** |

**Instruction:** وش معنى "حرب زبون"؟

**Response:** "حرب زبون" معناها حرب صعبة وشديدة، زي الناقة الزبون اللي تكون صعبة المراس وما تنقاد بسهولة.

**Verdict to confirm or overturn:** _______________


---

## 6. Summary of asks for Task 2

1. **Fix the `Thinking:` scaffold asymmetry** in the generator — emit it on both sides or
   neither. Highest priority; DPO data is not usable for training until this is resolved.
2. **Decide the `format_type` semantics** — response shape (project policy) or inherited
   chunk shape. This alone accounts for 3,484 of 4,143 rejections.
3. **Produce the other 8 rejection types**, especially `wrong_register` and
   `weak_organization`.
4. **Generate dialect records** — half the corpus (337 chunks, 5 regions) produced nothing.
5. **Confirm `accepted.jsonl` vs `candidates.jsonl`** — currently byte-identical — and
   delete `candidates copy.jsonl`.
6. **The 139 "inverted" pairs are NOT a confirmed defect** — see §2.1. No action asked
   for yet; we will re-measure after fixing our morphology handling and come back if a
   real inversion population survives.

## 7. What Task 3 owes

1. **Morphological handling in `check_facts.py`** before the fact axis is quoted again on
   conversational text (section 4). This blocks §2.1 and §2.3 as well as §4 — with
   similarity off, `check_dpo.direction` is a pure function of these verdicts, so the
   defect propagates undiluted into the DPO results.
2. **Re-run with similarity enabled.** These numbers are fact-axis only
   (`--no-similarity`); `sentence-transformers` is installed and the embedding axis has
   not yet been run at this scale.
3. **The LLM judge has not been run at all** — it needs an endpoint, and it needs
   rejection types other than `partial_factual_errors` to be worth running.

## Reproducing these numbers

```bash
python src/verification/adapt_task2.py \
    --sft-out data/adapted/sft_task2.jsonl --dpo-out data/adapted/dpo_task2.jsonl
python src/verification/check_facts.py --corpus classical_lexicon \
    --in data/adapted/sft_task2.jsonl
python src/verification/verify_sft.py  --corpus classical_lexicon \
    --in data/adapted/sft_task2.jsonl --no-similarity
python src/verification/check_dpo.py   --corpus classical_lexicon \
    --in data/adapted/dpo_task2.jsonl --no-similarity
```

The adapter never writes to `data/generated/**`; Task 2's files are untouched by anything
in this report.

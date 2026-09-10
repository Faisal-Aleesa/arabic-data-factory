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

**Post-fix (§8): 139 → 120.** The remaining 120 still rest on a fact axis whose extractor
half is unfixed, so this is still not a defect count.

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

**Post-fix (§8): 2,803 → 2,833 ties — it went UP.** With contradictions collapsing to
REVIEW on both sides, more pairs land on identical verdicts. Enabling the similarity
axis breaks only 5.3% of them, and 100% of the large-gap survivors are blocked by the
ungrounded guard (§9).

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

### 3.2 `NO_CHECKABLE_CLAIMS` — 3,333 records (72%) — now 467, see §10

`check_facts.py` found no checkable assertions in 72% of responses. This is a property of
the checker meeting a kind of text it was not built for — see section 4 — not evidence
that the responses are empty or wrong.

---

## 4. CAVEAT — `check_facts.py` was NOT validated on conversational paraphrase

> **STATUS 2026-09-08: diagnosed and FIXED. See §8 for the fix and its measured
> effect, and §9 for what it did NOT fix.** This section is kept as written because
> it is the diagnosis the fix rests on, and because the numbers in §2.1 and §2.3 were
> produced before it.

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
2. ~~**Assertion extraction for conversational text**~~ — **DONE for classical, §10**: NO_CHECKABLE_CLAIMS 3,333 → 467. Still unmeasured on dialect, and the 18% self-quoting artifact is an open refinement. Original note follows.
   This was the
   BIGGER of the two gaps: it produces the 3,333 `NO_CHECKABLE_CLAIMS` and it keeps the
   similarity guard shut on 2,092 pairs (§9). The morphology fix did not touch it.
3. **Similarity has now been run** at this scale (§9). It breaks 5.3% of ties and its own
   percentile distribution is saturated (median 98), so it is not a substitute for a
   working fact axis.
3. **The LLM judge has not been run at all** — it needs an endpoint, and it needs
   rejection types other than `partial_factual_errors` to be worth running.

---

## 8. The morphology fix — landed 2026-09-08

### What was wrong

`_relation()` classified any response value with characters welded onto a known fact as
`corruption`, which is correct for a multi-word NAME and wrong for a root. **Every
`entry_root` fact is a bare triliteral root — measured: 3,372 facts, all exactly 3
characters, 3,351 distinct, mean 8.5 per chunk.** Responses cite the inflected surface
form. `أثف` appears as `أثفية`, `مدد` as `مددت`, `جلح` as `الأجلح`. The generic rule read
ordinary Arabic derivation as textual corruption.

### The rule chosen, and why

Three candidates were measured against all 1,510 flagged `entry_root` assertions:

| rule | CONTRADICTED (n=715) | UNSUPPORTED (n=795) |
|---|---:|---:|
| **in-order subsequence** | **638 (89%)** | **253 (32%)** |
| contiguous substring | 513 (72%) | 0 (0%) |
| affix-strip then match | 429 (60%) | 0 (0%) |

Substring and affix-strip score 0% on UNSUPPORTED by construction — that set is defined by
containment having already failed. Only subsequence reaches it, and what it reaches are
hollow and defective roots whose letters are separated by infixed vowels.

**Specificity was measured, not assumed.** Tested against a RANDOM OTHER chunk's roots the
rule fires on **10 of 1,510 (1%)**. Of the tokens it does match, 797 match exactly one
root, 92 match two, 2 match three. It discriminates; it is not a rubber stamp. A
regression pin in `check_facts.run_self_test()` holds that property.

### The verdict is PARTIAL, not SUPPORTED

Deliberate. Naming a root the chunk covers is **not** the same as asserting the source's
claim about it — the response can still say something false. `PARTIAL` means "a human
should look", which is exactly the epistemic state. Marking these `SUPPORTED` would have
converted a false negative into a false positive.

### Measured effect

| record verdict | before | after |
|---|---:|---:|
| CONTRADICTED | 591 | **6** |
| UNSUPPORTED | 530 | 410 |
| PARTIAL | 147 | **852** |
| SUPPORTED | 44 | 44 |
| NO_CHECKABLE_CLAIMS | 3,333 | **3,333 (unchanged)** |

**980 assertions reclassified** — 709 from CONTRADICTED, 271 from UNSUPPORTED.

### Manual validation — 17 of 18

A fixed-seed random sample of the reclassifications was read individually. Clean
derivations included `ننقّب`←`نقب` (form II), `اخترص`←`خرص` (form VIII), `جمال`←`جمل`,
`معيشة`←`عيش`, `الجلحاء`←`جلح`, `شأفتهم`←`شأف`.

**One false positive: `أحمزها` matched root `حمأ`.** `أحمز` derives from `حمز`, not `حمأ`.
The match only succeeds because `norm_key` folds alef variants, so `حمأ` becomes `حما` and
the trailing `ها` supplies the final `ا`. **60 of 980 reclassifications (6%) depend on
that folding** and carry the same risk.

Left as-is rather than special-cased, because the destination is `PARTIAL` — a review
queue, not acceptance — and diverging from `norm_key`'s project-wide folding for one fact
type would cost more than it saves. Recorded here so the 6% is known rather than
discovered.

---

## 9. Why the ties do not break — the exact mechanism

This is the result most worth carrying forward, and it survived the fix.

### Similarity was run, and it does not rescue the ties

`check_dpo` was re-run over the same 3,000 pairs with the embedding axis ON:

| | fact axis only | with similarity |
|---|---:|---:|
| ties | 2,803 | 2,654 |

Paired per record: **149 of 2,803 ties broken (5.3%)**. And the breaks are weak evidence —
**88 favour chosen against 61 favouring rejected** (59%, z = 2.21, barely past chance), and
**40% of them sit at a percentile gap of 5–10**, i.e. just past the 5.0 tie margin.

### The mechanical reason

**724 still-tied pairs have a percentile gap of ≥10**, well past the tie margin. They
should have broken. They did not, and the cause is exact:

> **724 of 724 (100%) are ungrounded-versus-ungrounded.**
> `check_similarity.compare_percentiles()` returns `INCOMPARABLE_UNGROUNDED` and refuses
> to compare them.

That guard is correct and must not be relaxed — it exists because comparing two ungrounded
responses on similarity is comparing noise to noise, which previously flipped a verdict
between corpora on identical logic. Across all 3,000 pairs, 875 have a gap ≥10 and only
151 resolve to a direction.

The axis is also **saturated**: percentiles across all 6,000 responses run p10 = 51,
median = **98**, p90 = 100. Short answers about a chunk they genuinely derive from all look
alike. There is little spread left to discriminate with.

### The trap: groundedness is itself a fact-axis verdict

The guard keys on `is_grounded()`, which is `verdict != NO_FACT_COVERAGE`. So the fact axis
does not merely feed `direction` — **it also decides whether the independent axis is
allowed to speak at all.** A defect in fact checking silently disables the fallback that
would have compensated for it.

### The morphology fix did NOT un-suppress it — measured

| | before fix | after fix |
|---|---:|---:|
| pairs with both sides ungrounded | 2,092 (70%) | **2,092 (70%)** |
| un-suppressed | — | **0** |

**Zero.** The prediction that the fix would open the similarity axis was wrong, and the
reason is structural: `NO_FACT_COVERAGE` means the EXTRACTOR found no assertions at all.
The fix changes how existing assertions are *judged*; it cannot create assertions where
none were extracted. Chosen-side and rejected-side `NO_FACT_COVERAGE` counts are byte-for-
byte identical before and after (2,137 and 2,212).

**Two independent defects, and only one is fixed:**

| defect | module | status |
|---|---|---|
| assertions mis-judged (morphology) | `check_facts._relation()` | **FIXED** (§8) |
| no assertions found in conversational paraphrase | `extract_facts.py` | **CLOSED for classical** (§10); untested on dialect |

The extractor gap is what blocks the similarity guard AND what produces the 3,333
`NO_CHECKABLE_CLAIMS`. It is the larger of the two and is untouched.

### Post-fix DPO numbers (fact axis)

| | before fix | after fix |
|---|---:|---:|
| FLAG_SUSPICIOUS | 2,942 | 2,953 |
| NEEDS_JUDGE | 28 | 45 |
| AUTO_CONFIRM | 30 | **2** |
| ties | 2,803 | 2,833 |
| `rejected_better_verdict` | 139 | 120 |

Ties rose and inversions fell for the same reason: with contradictions collapsing to
REVIEW on both sides, more pairs land on identical verdicts. **AUTO_CONFIRM falling 30 → 2
is correct, not a regression** — those confirmations rested on `FAIL_CONTRADICTED` on the
rejected side, which the fix showed to be morphological false positives.


---

---

## 10. The extraction fix — landed 2026-09-09

§9 named `extract_facts.py` as the larger of the two open defects. This closes it for
classical data.

### What was wrong — structure, not pattern coverage

Every classical chunk rule anchors on dictionary-entry SHAPE. `CLA_ROOT` is
`^(root)\s*:` at a line start. A conversational answer never has that shape, so the
extractor found nothing in 3,333 of 4,645 records — not because the content is
unverifiable, but because it is verifiable in a different form.

Measured over those 3,333:

| feature | share |
|---|---:|
| quotes a lexical item (instruction **or** response) | **99%** |
| quotes it in the **instruction** | 98% |
| carries a definitional connective (`يعني`, `معناها`, `يقصدون`…) | 88% |
| quotes it in the **response** | 43% |

The shape is uniform: **the instruction quotes an item, the response glosses it.**
`ما هي 'جهمة الليل'؟` → *the last part of the night, near dawn*.

That 98/43 split forced an interface change. `check_response()` only ever received the
response, which caps coverage at 43% by construction. It now takes an optional
`instruction=None` — backward compatible, and a self-test pins that it stays callable
without it.

### Approaches considered, measured before building

| approach | coverage | precision | cross-chunk false hit |
|---|---:|---:|---:|
| A. quoted span, response only | 43% | 90% | 2% |
| **B. quoted span, instruction-preferred** | **99%** | **94%** | **1%** |
| B′. B, multi-word spans only | — | 87% | **0%** |
| **C. B + gloss overlap vs local chunk window** | **93%** | **84%** | 6% |
| D. content-word overlap, no quotes | 100% | 43% | 11% |

D was rejected: 43% against 11% is a ratio of roughly 4:1, versus about 94:1 for B. It
would manufacture assertions rather than find them.

B ships as the new fact type `quoted_lexical_item`; C ships as its judging rule. B alone
would have marked 3,290 records as carrying a checkable claim that essentially always
passes — worse than silence, because it looks like verification while detecting only a
fabricated headword.

**PARTIAL on success, never SUPPORTED.** 84%/6% is real discrimination but far weaker than
the morphology rule's 1% false-hit rate, and one shared content word is thin evidence that
a gloss is *correct*. A self-test asserts `quoted_lexical_item` can never reach SUPPORTED.

### Measured effect

| fact verdict | before §8 | after §8 (morphology) | after §10 (extraction) |
|---|---:|---:|---:|
| **NO_CHECKABLE_CLAIMS** | 3,333 | 3,333 | **467** |
| PARTIAL | 147 | 852 | 3,140 |
| UNSUPPORTED | 530 | 410 | 1,001 |
| SUPPORTED | 44 | 44 | 32 |
| CONTRADICTED | 591 | 6 | 5 |

**NO_CHECKABLE_CLAIMS falls 3,333 → 467, an 86% reduction.** 6,391 `quoted_lexical_item`
assertions were produced: 5,294 PARTIAL, 1,097 UNSUPPORTED.

Record-level verdicts: REVIEW 4,173 · NO_FACT_COVERAGE 467 · FAIL_CONTRADICTED 5.

### Manual validation — 16 of 18

Fixed-seed random sample, 16 distinct chunks, read individually. Correctly located and
glossed source idioms included `تأبط السيف`, `قرف العضاه`, `لهث الكلب`,
`أصفقوا على أمر واحد` and `شيبتني قوارع القرآن`.

**Two failures, both the same mechanism:** the model wrapped **its own paraphrase or a
coined example** in quotes — `'تمييز الذهب منه وتخليصه'`, `'أفول النجوم'` — and the
checker read it as a claim about the source. It is literally true that the source lacks
the phrase, and it is not evidence of hallucination, because no such claim was made.

### A double-extraction defect found by the self-test

Multi-word quoted spans were being extracted **twice**: as `entry_root` by the existing
`LEXICAL_MARKED` pattern, and as `quoted_lexical_item` by the new one. A record takes the
WORST of its assertions, so the bogus `entry_root` UNSUPPORTED masked the correct PARTIAL
from the very same span. Every classical `entry_root` fact is exactly 3 characters, so a
phrase could never match one — and §8's morphology fix could not reach it either, because
there is no root to inflect.

The first fix was too broad and broke four existing self-test cases: the DIALECT corpus's
`entry_headword` values ARE frequently multi-word (`DIA_HEADWORD` allows 40 characters
including spaces). The exclusion is now scoped to `entry_root` only. Worth recording
because the over-broad version passed a casual reading and only the existing cases caught
it.

### KNOWN LIMITATION — the self-quoting artifact (~18%)

Measured across all 1,097 UNSUPPORTED `quoted_lexical_item` assertions:

| where the quoted item appears | count | reading |
|---|---:|---|
| in the **instruction** | **905 (82%)** | a real claim the source does not support |
| **only in the response** | **192 (18%)** | the model quoting its own wording |

The extractor cannot tell "quoting the source" from "quoting my own paraphrase", because
both are just quoted spans.

**BUILT AND MEASURED 2026-09-09.** It is now `is_self_quote()` in `check_facts.py`.
Reading the 190 by hand first showed the table above is too clean: they are **not**
uniformly self-quotes. Genuine source claims sit among them — a hadith quotation, and the
construction "the man is-described-as X" — so the 18% was never a pure
artifact rate. (The Arabic is paraphrased here rather than quoted: the verbatim
phrasing occurs in the rights-pending corpus and the licence audit refuses it.)

The rule as finally built. An item is a SOURCE CLAIM (verdict unchanged) when **any** of:
- it is quoted in the instruction, or
- the instruction quotes nothing at all, or
- the response contains no definitional connective, or
- it occurs BEFORE the first connective.

Otherwise it is the model's own wording and softens to **PARTIAL**, never to SUPPORTED.

**The third condition was not in the proposal and is load-bearing.** Without it the rule
broke three existing specificity pins. The probe `يعني "خخخخ ذذذذ" هي عععع.` is
positionally *identical* to the real self-quote `يقصدون 'ما رأتك عيني منذ زمان'` —
connective at position 0, quoted span immediately after — and the two have opposite
ground truth. What separates them is that the real case has an instruction quoting the
source term, making the response's different span visibly a gloss. No instruction means
no evidence, and no evidence keeps the harsher verdict.

**Validated twice, and the second time on held-out data.** A 34-case hand-labelled sample
scored 88.9% precision / 91.7% recall and every one of its five misses was a coined
example introduced by a comparison marker (`زي`, `مثل`). Adding those markers fixed all
five — but that measurement was then fitted to the sample that diagnosed it, so the rule
was re-validated on a **fresh disjoint 22-case sample**:

| | precision | recall |
|---|---:|---:|
| as originally proposed, fitted sample | 88.9% | 91.7% |
| as built, **held-out sample** | **100%** | **81.8%** |

The instruction precondition trades recall for precision, which is the right direction
here: a false positive softens a *real* hallucination and hides it, while a false negative
merely leaves a reviewable UNSUPPORTED.

**Measured effect on the delivery**, not estimated:

| | before | after |
|---|---:|---:|
| `quoted_lexical_item` UNSUPPORTED assertions | 1,097 | **998** |
| record-level UNSUPPORTED | 1,001 | **944** |

99 assertions softened, moving 57 records. The original estimate of "on the order of 192"
was roughly double the real figure.

**Known limitations.** The connective is searched anywhere in the response, so an
unrelated earlier clause can trigger it — that is the single held-out false positive
(`زي الأكل والشرب` in a clause unconnected to the quoted item). Scoping the search to the
item's own sentence would probably fix it and is **not built**, because it is unmeasured.
The rule also only applies where the instruction quotes something, which is **79.7% of
these instructions** (3,703 of 4,645); on the rest it silently does nothing, and that
failure is safe by design.

### CAVEAT — validated on CLASSICAL data only

Every number in this section comes from `asas_albalagha` records. **There is nothing to
test the dialect side against: Task 2 produced zero dialect records** (394/394 classical,
0/337 dialect — §0).

Two specific reasons the dialect side may behave differently:

- **Quoting convention.** The 53% single-quote / 46% double-quote split, and the 99%
  quoting rate, are properties of this generator's classical prompt. A dialect prompt may
  quote less, or mark items with parentheses as the dialect CORPUS does.
- **`entry_headword` is not `entry_root`.** Dialect headwords are frequently multi-word,
  so the `entry_root` scoping above does not apply there, and a multi-word dialect span
  may legitimately match a headword rather than needing the quoted-item path at all.

**Re-measure coverage, precision and cross-chunk specificity on real dialect records
before quoting any of these figures for the dialect half.**


---

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

---

## 11. Duplication and leakage — landed 2026-09-09

Two Group D checkers now run against the adapted data:
`src/verification/check_duplication.py` (item 11, near-duplicate detection at the
SFT/DPO output level) and `src/verification/check_leakage.py` (train/val/test leakage
under `docs/SPLIT_POLICY.md`). Both reuse `data_engineering/dedup.py`'s MinHash/LSH
machinery rather than restating it, including its LSH-as-candidate-generator design.

Numbers below are from real runs on `data/adapted/` — **4,645 SFT records and 3,000 DPO
pairs** — not from fixtures.

### 11.1 Duplication: the data is clean, with two exceptions

| check | result |
|---|---|
| `EXACT_DUP` — byte-identical after whitespace collapse, any field | **0** |
| `NEAR_DUP` — Jaccard ≥ 0.80 within a field | **0** |
| `PAIR_COLLAPSE` — `rejected` is a near-copy of its own `chosen` | **20** (2 real, see below) |
| `CROSS_FILE` — DPO `chosen` is also an SFT `response` | **3,000 (100%)** |

Zero exact and zero near duplicates across all five text fields. That is a genuinely
clean result, and it was checked rather than assumed: the maximum pairwise Jaccard among
all 4,645 SFT responses is **0.385**, so nothing sits anywhere near the threshold, and an
injected copy is still caught (self-test case 2–3).

**CORRECTED 2026-09-09 — the first version of this section over-claimed.** It said all
19 pairs "carry no preference signal at all". That is true of 2 of them and **wrong about
the other 17**. High lexical overlap is not the same as absence of signal, and a Jaccard
threshold cannot tell "the two sides say the same thing" from "one word changed and the
meaning inverted".

The count is now **20** rather than 19 (the adapter fix in §12 unmasked one more), and
the word-level diff decomposes them cleanly:

| | count | reading |
|---|---:|---|
| byte-identical `chosen` and `rejected` | **2** | zero preference signal — **real defects** |
| exactly one edited region | **18** | valid `partial_factual_errors` pairs |

The 18 are single-token factual substitutions, which is precisely what that rejection
type is supposed to produce:

| chosen → rejected | |
|---|---|
| `نهايته` → `بدايته` | its end → its beginning |
| `الدرهمين` → `الريالين` | two dirhams → two riyals |
| `واليمن` → `والشام` | Yemen → the Levant |
| `الكبير الواسع` → `الصغير` | the large and wide → the small |
| `الناقة الشابة` → `الجمل الصغير` | the young she-camel → the small camel |

Minimal edit, inverted meaning. That is **high-quality preference data** — arguably the
most valuable kind, because it forces the model to attend to the fact rather than the
style. Reading them as defects would have been an argument for regenerating good data.

All 20 are `rejection_type: partial_factual_errors`, `dictionary_entry`, `classical`, one
model version, spread across 16 distinct chunks at file positions 29 to 2917 — so not a
contiguous batch failure.

**`PAIR_COLLAPSE` is therefore a triage signal, not a defect count.** It has a ~10% true
defect rate on this delivery. Only comparing the two sides of one record surfaces these
at all, which is still worth doing — but the finding must be read per-pair.

Full distribution of `jaccard(chosen, rejected)`: median 0.067, p90 0.407, max 1.000.

| threshold | pairs | share |
|---|---:|---:|
| ≥ 0.95 | 2 | 0.1% |
| ≥ 0.90 | 5 | 0.2% |
| ≥ 0.80 | 19 | 0.6% |
| ≥ 0.70 | 70 | 2.3% |
| ≥ 0.60 | 149 | 5.0% |

The distribution is included because 0.80 is a threshold to argue with, not a fact. The
70 pairs at ≥ 0.70 are a weak training signal even though they are not reported.

**Ask for Task 2:** regenerate the **2 byte-identical pairs** (file indices 45 and 1540).
The other 18 need no action — they are good preference data.

### 11.2 The one parameter that had to change, and why

`dedup.py` uses `SHINGLE_SIZE = 5`, tuned for whole documents of thousands of words.
These records are short — SFT responses median **17** words, instructions median **6**.
`shingles()` returns the whole text as a single shingle when the text is shorter than the
shingle size, so at size 5 that affects **27.2% of SFT instructions** (1,262 of 4,645),
with 64.0% producing fewer than three shingles. For that quarter of the field the
near-duplicate check silently degrades into an exact-match check and then reports a clean
near-duplicate result it never performed.

`check_duplication.py` uses size 3, where the degenerate share is 0.0%. Both sizes were
run against the real data and both return zero near-duplicate pairs, so the change did not
manufacture the verdict — it makes the check capable of producing one. The per-field
`degenerate_texts` count is in every report so a shorter future dataset cannot hide the
same way.

**Limitation, stated plainly.** On text this short, a Jaccard threshold detects *copies*,
not *paraphrases*. Measured: at 20 words a 3-word tail edit already drops similarity to
0.68, below the 0.80 threshold. Do not read "0 near-duplicates" as "no redundant content".

### 11.3 Leakage: no split is possible at all

`check_leakage.py` in feasibility mode, against the adapted data:

```
splittable units (parent documents): 1
  asas_albalagha    7,645 records   (100.0% of all records)
two-way split IMPOSSIBLE, three-way split IMPOSSIBLE
```

Every adapted record derives from `asas_albalagha`. `SPLIT_POLICY.md` requires splitting
by parent document, and there is exactly **one** parent document, so no train/val/test
split of this delivery conforms to the policy. This is the direct downstream consequence
of the delivery containing **zero dialect records** (§3): the missing dialect coverage is
not only a distribution problem, it removes every splittable unit but one.

Feasibility mode exits **2**, never 0, and prints "no partition check was performed."
A run that checked nothing must not read as a pass.

### 11.4 The naive split fails loudly, as it should

To confirm the checker detects what it claims, the split a team would actually reach for
first — SFT as train, DPO as test — was run through VERIFY mode:

```
DOC_IN_MULTIPLE_PARTITIONS     1
TEXT_CROSSES_PARTITIONS    6,002      (5,485 exact, 517 near)
exit 1 — FAIL
```

Of the 6,002 crossings, **2,983 involve text of 10 or more tokens**, so they cannot be
dismissed as short generic phrases (828 are ≤ 4 tokens and probably can be). This is the
cross-file overlap from §11.1 doing exactly what it threatens: the same sentence lands in
train and test while the document rule is still satisfied on paper.

Those counts were measured before the §12 adapter fix, when the overlap read 82.8%. The
true overlap is **100%**, so the crossing count is a floor, not a ceiling. The verdict —
exit 1, FAIL — does not change, and the finding is strictly stronger.

**The document check alone would not catch this**, which is why the text check exists.

### 11.5 A defect the checker found on its first run — since FIXED

All 6 Najdi chunks reported `UNKNOWN_DOC_ID`. The licence manifest tracked that corpus as
`dialect_dict_najdi_popular`; its chunk ids carry `majam_alkalimat_alshaabia_najd`.
`SPLIT_POLICY.md` had asserted that the manifest's `doc_id` set and the corpus's are
identical, so that the split unit and the licence-tracked unit are the same thing. That
invariant had broken.

**Fixed 2026-09-09** by renaming the manifest row to `majam_alkalimat_alshaabia_najd`, so
the manifest follows the data. The alternative — re-emitting the chunks under the
manifest's name — was rejected because those chunk ids are already committed and pushed,
whereas the manifest value was verified to be referenced nowhere but that row and prose
comments. Re-run against the real chunk ids: `UNKNOWN_DOC_ID 0`, exit 0, PASS.

The cost is that this row alone does not follow the `dialect_dict_*` naming convention of
the other five dialect sources. The convention is cosmetic; the invariant is load-bearing.

This is the checker paying for itself on day one: the mismatch was invisible to every
other check in the repo, and would have surfaced as a licence-tracking gap only once
records began deriving from that corpus.

### 11.6 A correction that changed two of the numbers above

Sections 11.1 and 11.4 originally reported the cross-file overlap as **2,485 of 3,000
(82.8%)**. The true figure is **3,000 of 3,000 (100%)** — every DPO `chosen` is an SFT
`response`. The 515 that looked distinct were masked by a seven-character `Answer:`
prefix our own adapter failed to strip; see §12. This makes the leakage finding stronger,
not weaker: splitting the SFT and DPO files independently leaks **every single pair**,
not most of them.

### 11.7 What is NOT verified

- No partition check has been performed on real partitions, because none exist. §11.4 is
  a deliberately constructed failing case, not a released split.
- Near-duplicate sensitivity is a copy check, not a paraphrase check (§11.2).
- `canonical_text` collapses whitespace but does **not** fold alef/ya. Two spellings of
  one word are two different shingles, so a spelling-variant near-duplicate is invisible.
  Folding was rejected on `clean.py`'s reasoning: the fold is right for matching and wrong
  for text you keep, and a duplicate verdict computed on a form that is not the shipped
  form would describe a text that does not exist.
- Near-duplicates are **flagged, not deleted**, per the standing project rule. Removal is
  opt-in via `--apply`, which writes a new file and never edits in place.

### A note on reading the reports

`duplication_report.json` **samples** its per-finding lists at 50 entries. The COUNTS are
computed before the cap and are never reduced — `cross_file_findings_total: 2485` sits
next to a 50-entry `cross_file_findings` list, with `cross_file_findings_truncated: 2435`.
Enumerating all 2,485 produced a 17,712-line, 466KB file in a repo where every other
report is 21–62 lines, and it would churn entirely on each regeneration. The full lists
are still returned in memory, so `--apply` and any programmatic caller see everything.

### Reproducing §11

```bash
python src/verification/check_duplication.py --self-test
python src/verification/check_duplication.py
python src/verification/check_leakage.py --self-test
python src/verification/check_leakage.py            # feasibility; exits 2 by design
```

---

## 12. URGENT, and it was OURS — the adapter left 17.2% of the scaffold in place

**Fixed 2026-09-09.** This is a defect in `src/verification/adapt_task2.py`, not in
Task 2's output. It is recorded here because it corrupted numbers this document reported
to Task 2, and because it is a live instance of the §1 scaffold confound that the adapter
exists to remove.

### What was wrong

The scaffold has **two** shapes. The full form is `Thinking:\n...\n\nAnswer:\n...`; the
short form is a bare `Answer:` header with no Thinking section. The stripping pattern
required `Thinking:` to be present, so bare-header records passed through untouched.

The two shapes partition the file exactly:

| shape | records | stripped before the fix? |
|---|---:|---|
| full `Thinking:` + `Answer:` | 2,485 | yes |
| bare `Answer:` header | **515 (17.2%)** | **no** |
| total | 3,000 | |

So 515 `chosen` values shipped beginning with `Answer:\n` while `rejected` carried that
token in **zero** records. That is exactly the asymmetry §1 flags as making DPO
unmeasurable: a preference model can learn "prefer the text starting with `Answer:`" and
score well without reading any Arabic. Our mitigation for the confound was itself
reproducing it, at a sixth of the corpus.

### It also hid its own consequences

The residual seven-character prefix made those 515 `chosen` values compare unequal to
their byte-identical SFT `response` twins:

| | measured | true |
|---|---:|---:|
| DPO `chosen` that is also an SFT `response` | 2,485 (82.8%) | **3,000 (100%)** |

All 515 match once the prefix is removed. **Every DPO `chosen` is an SFT `response`** —
so splitting the two files independently leaks every pair, not 83% of them.

A stripping bug that also suppresses the measurement of itself is the worst shape this
class of defect takes, and it is why `run_self_test()` now pins **both** scaffold shapes,
four whitespace variants of the bare header, and two discrimination cases (a mid-text
`Answer:` and the word `Answers:`) that must **not** be stripped.

### A counter that could not reveal the bug

`scaffold_in_chosen` was derived from `thinking is not None`, so it under-reported by
exactly the 515 bare-header records — the summary said 2,485 stripped when the true figure
is 3,000. A counter that misses the same cases the stripper misses cannot expose the
stripper. It is now derived from whether anything was actually removed.

### What changed downstream

| measurement | before | after |
|---|---:|---:|
| `CROSS_FILE` overlap | 2,485 (82.8%) | **3,000 (100%)** |
| `PAIR_COLLAPSE` | 19 | **20** |
| `quoted_lexical_item` verdicts | unchanged | unchanged |
| leakage: splittable units | 1 | 1 |

`quoted_lexical_item` is unaffected because the residual prefix only ever sat on DPO
`chosen`, and that check runs on SFT `response` values. The leakage conclusion is
unchanged and strictly stronger.

**Task 2 needs to know the corrected overlap figure**, since it raises the urgency of the
§1 scaffold issue rather than lowering it.

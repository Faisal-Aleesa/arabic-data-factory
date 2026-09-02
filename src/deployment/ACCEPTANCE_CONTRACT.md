# SFT acceptance contract — read before writing the release stage

**عقد القبول: قبل كتابة مرحلة الإصدار.**

This package is empty by design. When the stage that decides which SFT records reach the
released dataset gets built — here, or in `src/dpo/`, or wherever — it must follow the
rule below.

---

## The rule

> **Never accept a record on `verdict` alone.** A verification verdict answers exactly one
> question: *is this response grounded in its source?* It says nothing about whether the
> response is shaped the way its `format_type` promises.
>
> A grounded-but-malformed response carries a `PASS` verdict. Accepting on the verdict
> alone lets it through silently.

Call the sanctioned predicate:

```python
import verify_sft as vs

result = vs.verify_sft_record(record, ctx)      # NOT verify_response()
decision, reason = vs.acceptance_decision(result)
# decision is ACCEPT / REJECT / HOLD_FOR_REVIEW
```

`acceptance_decision()` consults both axes and is the only supported way to turn a
verification result into an accept/reject.

## Why it is a function and not a documented convention

The same reasoning as `check_similarity.compare_percentiles()`: a rule that depends on
everyone remembering it will eventually be forgotten by someone reading `result['verdict']`
and drawing the obvious conclusion. So the requirement is enforced by the signature.

- It takes a **`verify_sft_record()`** result. `verify_response()` runs no format check, so
  its output has no format fields, and passing one raises `ValueError` rather than
  defaulting to anything. **A missing check must never resolve to acceptance.**
- `format_validated` false → `HOLD_FOR_REVIEW`, never `ACCEPT`. An unchecked format is not
  a clean format.
- `format_conforms` false → `REJECT`, whatever the verdict says. This is the specific case
  the contract exists to prevent.

## Decision table

| condition | decision |
|---|---|
| `INVALID_RECORD` | REJECT — the record is unusable |
| `FAIL_CONTRADICTED` | REJECT — a precise fact conflicts with the source |
| `format_validated` false | HOLD_FOR_REVIEW — no definite format answer |
| `format_conforms` false, `dictionary_entry` | **REJECT — grounded but malformed** |
| `format_conforms` false, any other format | HOLD_FOR_REVIEW — see tiers below |
| `PASS` + conforms | ACCEPT |
| `REVIEW` / `NO_FACT_COVERAGE` + conforms | HOLD_FOR_REVIEW — needs a human or the judge |

The self-test in `verify_sft.py` pins every row of this table, plus three invariants: a
`PASS` verdict is never accepted without a definite conforming format answer; a bare
`verify_response()` result (no format fields) is refused rather than decided; and every
format `check_format` knows sits in exactly one tier.

## Two tiers, because the evidence is not equal

A `format_conforms: false` finding is only as strong as the check behind it, and the
checks are not equally validated.

| tier | format_types | on mismatch |
|---|---|---|
| validated against real corpus content | `dictionary_entry` | **REJECT** |
| validated on synthetic cases only | `prose`, `narrative_paragraph`, `verse`, `list`, `footnote_block` | HOLD_FOR_REVIEW |

**Measured:** both corpora are 100% `dictionary_entry` (337 + 394 chunks), because both
sources were ingested with `--format dictionary`. That is the only format whose check has
ever met real data. The other five are implemented against `chunk.py`'s own assignment
rules and exercised only by hand-written cases, so a mismatch there is weaker evidence —
weak enough that discarding a record on it would be discarding it on a check that has
never been tested against real data of its kind.

The tiers live in `verify_sft.FORMAT_HARD_REJECT` / `FORMAT_REVIEW_ONLY`. **When a corpus
in one of those formats is ingested and its check is validated against real content, move
that `format_type` into the hard-reject set** and record why here. A self-test asserts
every format `check_format` knows sits in exactly one tier, so adding a format without
deciding its tier fails loudly.

## Assumption the gate depends on: `format_type` describes the RESPONSE

**Project policy (communicated to the Task 2 / reconstruction team):** a record's
`format_type` states the shape the response is *intended* to have. It is **not** an
inherited copy of the source chunk's format.

**سياسة المشروع: يصف حقل `format_type` شكل الاستجابة المقصود، لا شكل المقطع المصدر.**

This matters because the whole gate rests on it. If reconstruction copies `format_type`
from the chunk, then a summarisation instruction over a `dictionary_entry` chunk produces
flowing prose labelled `dictionary_entry`, every such record mismatches, and the gate
rejects work that is perfectly good. Mismatches would become expected noise, and the
correct response would be to weaken the gate — which would then stop catching the real
defect it exists for.

Because the policy holds, a mismatch means what it says: **the response is not the shape
it claims to be.** That is a real defect, not noise, and `dictionary_entry` mismatches are
rejected accordingly.

If reconstruction ever cannot honour this — if `format_type` must for some reason mirror
the chunk — then this gate's `dictionary_entry` tier has to be revisited *before* that
change ships, not after. It is the assumption that makes hard rejection defensible.

## What this does NOT settle

- **Format is reported, not gated, inside `verify_sft`.** The verdict stays a pure
  grounding signal on purpose: a well-grounded response in the wrong shape is a different
  defect from a fluent invention, and merging them makes both harder to read. The gate
  lives at acceptance, which is where a pass/reject actually happens.
- **No thresholds are settled.** `SimilarityBands` defaults are provisional, derived from
  12 hand-written probes, and real reconstructed output should decide them. See
  `check_similarity.py`.
- **DPO pairs are a separate path.** `check_dpo.py` has its own triage
  (`AUTO_CONFIRM` / `NEEDS_JUDGE` / `FLAG_SUSPICIOUS`) and already consults format for its
  `wrong_formatting` type. This contract governs the SFT record path.

## Layered by design

Automated checks are one layer, not the final word. `HOLD_FOR_REVIEW` is the expected
outcome for a large share of records and should route to the judge or a human — it is not
a failure state to be optimised away by loosening the rules above.

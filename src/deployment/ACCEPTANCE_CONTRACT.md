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
| `format_conforms` false | **REJECT — grounded but malformed** |
| `PASS` + conforms | ACCEPT |
| `REVIEW` / `NO_FACT_COVERAGE` + conforms | HOLD_FOR_REVIEW — needs a human or the judge |

Ten self-test cases in `verify_sft.py` pin this table, including the invariant that a
`PASS` verdict is never accepted without a definite, conforming format answer.

## What this does NOT settle

- **Format is reported, not gated, inside `verify_sft`.** The verdict stays a pure
  grounding signal on purpose: a well-grounded response in the wrong shape is a different
  defect from a fluent invention, and merging them makes both harder to read. The gate
  lives at acceptance, which is where a pass/reject actually happens.
- **`check_format` confidence is uneven.** Both corpora are 100% `dictionary_entry`, so
  that is the only `format_type` validated against real corpus content. `prose`, `verse`,
  `list` and `footnote_block` rest on `chunk.py`'s own assignment rules and synthetic
  cases. A `format_conforms: false` on those is weaker evidence than on a dictionary
  entry, and the release stage may reasonably route them to review instead of rejecting.
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

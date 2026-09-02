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

## Judge vs rule-based disagreement

**⚠️ POLICY, NOT YET CODE.** `acceptance_decision()` does not implement any of this: no
judge exists to produce a verdict, so there is nothing to test the interaction against.
Do not assume the code enforces it. When real judge output exists, implement the table
below and add self-test cases for every cell before wiring it into acceptance.

**سياسة، لم تُبرمج بعد.** لا وجود لحَكَمٍ يُنتج حكمًا حتى الآن، فلا شيء يُختبر عليه
التفاعل. لا تفترض أن الشيفرة تطبّق هذا.

The two layers answer different questions and are not interchangeable. The rule-based
layer checks verifiable propositions against the source: does this year match, is this
name the one the source carries, does this response conform to its declared format. The
judge assesses qualities no rule here can reach — tone, register, completeness,
naturalness, whether the answer actually serves the instruction.

Neither is superior in general. Each is authoritative in the region the other cannot see,
and the policy follows from that rather than from any ranking between them.

| rule-based | judge | outcome |
|---|---|---|
| `FAIL_CONTRADICTED` | anything, including PASS | **REJECT** — rules win outright |
| `PASS` / `AUTO_CONFIRM` | FAIL | **HOLD_FOR_REVIEW** — never an automatic reject |
| `PASS` / `AUTO_CONFIRM` | PASS | ACCEPT |
| any `REVIEW` / `HOLD_FOR_REVIEW` | anything | **HOLD_FOR_REVIEW** |
| anything | any judge REVIEW / abstain | **HOLD_FOR_REVIEW** |

### 1. A rule-based `FAIL_CONTRADICTED` overrides any judge verdict

A confirmed factual contradiction is not overridable by an overall quality impression.
`FAIL_CONTRADICTED` means a *precise* fact in the response conflicts with the source: a
year that does not match, a name the source does not carry. That is a verifiable
proposition, checked against the actual chunk, and it does not become less true because
a response reads well.

This is the exact failure the whole verification stack exists to catch — the dangerous
response is the one that looks grounded and is not. A fluent, well-organised answer with
one wrong date is *more* dangerous than an obviously poor one, because it survives
casual review. A judge scoring holistic quality is precisely the reader most likely to
pass it.

Note the rule is narrow on purpose. It applies to `FAIL_CONTRADICTED`, which
`verify_sft` already restricts to precise fact types — a region-only contradiction
routes to REVIEW rather than hard failure, because it rests on a heuristic over a lossy
field. The override inherits that narrowness and must not be widened to cover it.

### 2. A judge `FAIL` over a rule-based `PASS` routes to review, not rejection

The rule-based layer is structurally blind to whole categories of defect. It has no
register classifier, no organisation checker, no measure of whether an answer is
complete or reads naturally — those gaps are documented, measured and listed in
`check_dpo.py`'s coverage table, where 2 of 9 charter weakness types are invisible to
every automated check here and several more only incidentally visible.

So a judge failing something the rules passed is the expected and useful case: it is
probably seeing a real defect in the region the rules cannot reach. Auto-rejecting on it
would still be wrong, because the judge is a model and can be mistaken, inconsistent
between runs, or reacting to style rather than substance. Review is where a
disagreement between two partial views belongs.

This asymmetry with rule 1 is deliberate and worth stating plainly: **rules override the
judge, but the judge does not override the rules.** Not because rules are better, but
because a rule-based failure is a checkable claim about the source while a judge failure
is an assessment. A checkable claim can be confirmed by a human in seconds; an
assessment needs one.

### 3. Any REVIEW or HOLD from either layer keeps the record in review

Neither layer can clear the other's uncertainty. A judge `PASS` does not resolve a
rule-based `REVIEW`, and rule-based confidence does not resolve a judge's hesitation —
they are uncertain about different things.

This is the same principle already applied inside `verify_sft`: `NO_FACT_COVERAGE` is
neither a pass nor a fail because the fact axis was silent, not satisfied. Letting one
layer's confidence overwrite the other's uncertainty would launder a gap in coverage
into an endorsement.

Expect this to be the common outcome. `HOLD_FOR_REVIEW` is not a failure state to be
optimised away by loosening the rules — it is the layered design working.

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
- **The judge interaction is policy only.** The disagreement table above is NOT
  implemented in `acceptance_decision()`, which currently knows nothing about a judge.
  It cannot be implemented responsibly until real judge output exists to test the
  interaction against — every threshold and tier in this project that was set before
  meeting real data had to be corrected afterwards, and there is no reason to expect this
  one to be different.

## Layered by design

Automated checks are one layer, not the final word. `HOLD_FOR_REVIEW` is the expected
outcome for a large share of records and should route to the judge or a human — it is not
a failure state to be optimised away by loosening the rules above.

# DPO Data Quality Fixes

## Overview

This document describes critical issues found in DPO pair generation and the fixes implemented.

**Status:** ✅ All fixes implemented and ready to apply

---

## Issues Fixed

### Issue 1: Scaffold Contamination in `chosen` Values

**Problem:** DPO `chosen` values included the full reasoning scaffold:
```
Thinking: The word means ...
Answer: The answer is ...
```

**Impact:**
- 100% of DPO pairs had identical structure in `chosen` (scaffold present)
- 0% of `rejected` values had scaffold
- Model learns "prefer text with Thinking+Answer structure" → spurious signal
- 515 records with bare `Answer:` header created additional confound

**Fix:** Extract only the answer portion before generating DPO pairs
```python
from sft.generator import extract_answer_without_scaffold

clean_chosen = extract_answer_without_scaffold(full_response)
# Input: "Thinking: ... Answer: The actual answer"
# Output: "The actual answer"
```

**Status:** ✅ Implemented in `arabic_sft_dpo_pipeline.ipynb` cell-12

---

### Issue 2: Identical Chosen/Rejected Pairs

**Problem:** 2 pairs (indices 45, 1540) have identical `chosen` and `rejected` values
- No preference signal
- Waste training capacity
- Originally misidentified as 19 pairs (other 17 were single-token factual differences - those are valid)

**Fix:** Regenerate these 2 pairs with different rejection types
```bash
# Identify regeneration targets
python -m sft.fix_dpo_issues <dpo_file> <output_file>

# Then manually regenerate indices 45, 1540 with different rejection types
```

**Regeneration Strategy:**
- Use `wrong_register` for one (different tone/formality)
- Use `weak_organization` for other (reorder/confusing structure)
- Keep Saudi Arabic dialect
- Maintain semantic plausibility

**Status:** ⚠️ Requires manual regeneration

---

### Issue 3: Single Document Architectural Constraint

**Problem:** All 3,000 records derive from single source (`asas_albalagha`)
- Cannot create conforming train/val/test splits
- Any split leaks pairs (same sentence in train and test)
- Violates `docs/SPLIT_POLICY.md`

**Fix:** Acknowledge and document limitation
```markdown
# Data Source Constraint

All 3,000 SFT/DPO records derive from asas_albalagha.
- Single document source
- No document-level train/val/test partition possible
- Splitting requires text-level cross-partition checking

## Workaround

Use temporal or random split with text-level deduplication:
1. Random split (50/25/25)
2. Check for identical text across partitions
3. Move duplicates to prevent leakage
```

**Status:** ✅ Documented

---

## Implementation Details

### Scaffold Extraction Function

Location: `src/sft/generator.py`

```python
def extract_answer_without_scaffold(text: str) -> str:
    """
    Extract answer without thinking scaffold.
    Handles:
    - "Thinking: ... Answer: ..." → returns answer only
    - "Answer: ..." → returns text after header
    - Plain text → returns as-is
    """
```

### DPO Cleaning Script

Location: `src/sft/fix_dpo_issues.py`

```bash
python -m sft.fix_dpo_issues <input_dpo> <output_dpo>
```

**Output:**
- Cleaned JSONL file (no scaffold in chosen)
- Report of identical pairs found
- Stats on cleaning operation

---

## Before/After Examples

### Before (Contaminated)

```json
{
  "pair_id": "sample_001_dpo_00",
  "prompt": [{"role": "user", "content": "وش معنى خوص؟"}],
  "chosen": [{"role": "assistant", "content": "Thinking: خوص يعني أوراق النخل. Answer: أوراق النخل"}],
  "rejected": [{"role": "assistant", "content": "أوراق كثيفة"}],
  "rejection_type": "partial_factual_errors"
}
```

**Issues:**
- ✗ Chosen has scaffold, rejected doesn't
- ✗ Model learns structure preference
- ✗ Data contamination

### After (Clean)

```json
{
  "pair_id": "sample_001_dpo_00",
  "prompt": [{"role": "user", "content": "وش معنى خوص؟"}],
  "chosen": [{"role": "assistant", "content": "أوراق النخل"}],
  "rejected": [{"role": "assistant", "content": "أوراق كثيفة"}],
  "rejection_type": "partial_factual_errors"
}
```

**Benefits:**
- ✓ Both chosen/rejected are plain answers
- ✓ No structural confound
- ✓ Fair preference signal
- ✓ No data contamination

---

## Application Instructions

### Step 1: Update Notebook

The scaffold extraction is already implemented in:
- `notebooks/arabic_sft_dpo_pipeline.ipynb` (cell-12)

Just regenerate DPO pairs using the updated notebook.

### Step 2: Clean Existing Pairs (Optional)

If you have existing DPO data:

```bash
cd /path/to/project
python -c "
from src.sft.fix_dpo_issues import clean_dpo_dataset
stats = clean_dpo_dataset('data/generated/dpo/candidates.jsonl', 'data/generated/dpo/candidates_cleaned.jsonl')
print(stats)
"
```

### Step 3: Identify Regeneration Targets

```bash
python -c "
from src.sft.fix_dpo_issues import identify_regeneration_targets
targets = identify_regeneration_targets('data/generated/dpo/candidates_cleaned.jsonl')
for t in targets:
    print(f'Index {t[\"index\"]}: {t[\"pair_id\"]} - {t[\"reason\"]}')
"
```

### Step 4: Regenerate Problem Pairs

Manually regenerate the identified pairs with different rejection types.

---

## Validation Checklist

After applying fixes, verify:

- [ ] No `Thinking:` or `Answer:` headers in `chosen` values
- [ ] `chosen` and `rejected` never identical
- [ ] Both `chosen` and `rejected` are plain answer text
- [ ] Saudi Arabic dialect maintained in all values
- [ ] No structural confounds (formatting differences)
- [ ] All 3,000+ pairs have valid `rejection_type`
- [ ] File size ~similar to before (cleaning shouldn't reduce count)

---

## Related Documents

- `docs/SPLIT_POLICY.md` - Data splitting strategy
- `docs/TASK2_INTAKE_FINDINGS.md` - Original issue report
- `src/sft/generator.py` - Scaffold extraction implementation
- `src/sft/fix_dpo_issues.py` - Cleaning and validation script

---

## References

**Contamination Report:** "DPO/SFT overlap is 100%, not 82.8%"
- DPO chosen/SFT response overlap: 100% (complete contamination)
- Scaffold confound: 515 records (bare `Answer:` format)
- Identical pairs: 2 pairs (indices 45, 1540)
- Single document source: All from asas_albalagha

**Why This Matters:**
1. Structural bias renders preference training unreliable
2. Data contamination invalidates model evaluation
3. Train/test leakage makes generalization impossible

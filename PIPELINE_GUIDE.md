# SFT & DPO Pipeline Guide

## Overview

This is a rebuilt, well-structured pipeline for generating high-quality Arabic SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) training data.

**Location:** `notebooks/sft_dpo_pipeline_v2.ipynb`

**Core Modules:** `src/sft/`
- `schema.py` - Data structures with full validation
- `prompts.py` - Detailed system and user prompts
- `generator.py` - OpenRouter API integration
- `validation.py` - Schema and quality checking

## What Changed

### Before (Old Notebook)
- ❌ Unstructured, mixed concerns
- ❌ Weak prompts without detailed spec
- ❌ Limited quality metrics
- ❌ No proper schema validation
- ❌ No diversity checks

### After (New Pipeline)
- ✅ Clean 11-section notebook
- ✅ Detailed prompts matching your spec exactly
- ✅ Comprehensive quality scoring
- ✅ Strong schema validation
- ✅ Proper error handling and progress tracking

## Pipeline Stages

### 1. Setup & Configuration
Initialize project, load environment, configure API key and paths.

### 2. Chunk Inspection
Examine the input data structure and format.

### 3. Smoke Test
Test generation on 2 random chunks. No data saved.
- Shows sample Q/A/Thinking
- Validates prompt behavior
- **Run this before full generation**

### 4. SFT Generation
Generate question-answer pairs from all chunks using Gemini via OpenRouter.
- Calls model with detailed system prompt
- Falls back to simplified prompt if needed
- Progress tracking with chunk IDs
- Respects max_examples_per_chunk ceiling

### 5. SFT Validation
- Schema contract validation
- Quality metric computation
- Pass rate analysis (confidence >= 0.6)

### 6. Save SFT Candidates
Write samples to JSONL for review and verification.

### 7. Review Samples
Inspect random SFT samples before acceptance.

### 8. Accept SFT Samples
Mark high-quality samples (confidence >= 0.7) as ACCEPTED.
- Ready for DPO generation
- **In production:** Run external verification script here

### 9. Generate DPO
Create preference pairs: chosen answer + plausible rejected answer.
- Uses accepted SFT samples
- One meaningful flaw per rejected answer
- Rejection types tracked

### 10. Save DPO Candidates
Write DPO pairs to JSONL.

### 11. Summary
Report totals and next steps.

## Configuration (Edit Before Each Run)

In section 1.2, adjust:
```python
SFT_CHUNK_LIMIT = 394          # Number of chunks to process
DPO_SFT_LIMIT = 394            # Number of accepted SFT for DPO
MAX_EXAMPLES_PER_CHUNK = 12    # Safety ceiling
PAUSE_BETWEEN_CALLS = 0.5      # API call delay
```

## Quality Metrics

Each sample is scored on:
- **Question Naturalness** - Does it sound like natural Saudi Arabic?
- **Answer Completeness** - Does it fully address the question?
- **Instruction Quality** - Is the question clear?
- **Support Coverage** - Is answer grounded in support facts?
- **Thinking Quality** - Is reasoning semantic, not meta?
- **Saudi Dialect Quality** - Appropriate use of Saudi dialect
- **Information Density** - Useful info per word
- **Overall Teacher Confidence** - Weighted composite

Pass rate: Samples with confidence >= 0.6 are considered passing.

## Data Model

### SFTSample
- `sample_id` - Unique identifier
- `question` - Saudi Arabic question
- `thinking` - Semantic reasoning (0–200 words)
- `answer` - Saudi Arabic answer
- `messages` - [user_message, assistant_message] pair
- `question_type` - Definition/Meaning/Explanation/etc.
- `difficulty` - Easy/Medium/Hard
- `reasoning_mode` - Direct Recall/Contextual/etc.
- `reasoning_depth` - 0 (direct) to 3 (deep)
- `teacher_only` - Support facts, knowledge units, etc. (excluded from model input)
- `validation_status` - Unverified/Accepted/Review/Rejected
- `quality_metrics` - Comprehensive scoring

### DPO Pair
- `pair_id` - Unique identifier
- `source_sample_id` - Original SFT sample
- `prompt` - User question (unchanged)
- `chosen` - Accepted SFT answer
- `rejected` - Model-generated flawed answer
- `rejection_type` - Type of flaw introduced
- `verification_status` - Unverified/Accepted/etc.

## Output Files

```
data/generated/
├── sft/
│   ├── candidates.jsonl    # All generated SFT samples
│   └── accepted.jsonl      # Verified and accepted samples
└── dpo/
    └── candidates.jsonl    # DPO preference pairs
```

Each line is a JSON sample.

## Question Types Supported

- Definition
- Meaning
- Explanation
- Clarification
- Comparison
- Distinction
- Cause / Why
- Reasoning
- Inference
- Application
- Example
- Contextual Meaning
- Relationship
- Multi-step
- Conceptual

## Reasoning Modes

- Direct Recall - Pure memory retrieval
- Contextual Interpretation - Understanding via context
- Definition Resolution - Matching to definitions
- Disambiguation - Resolving multiple meanings
- Comparison - Analyzing differences
- Cause → Effect - Causation
- Evidence → Conclusion - Logical derivation
- Concept → Application - Applying concepts
- Multi-step Inference - Multiple logical steps
- Relationship Resolution - How things relate

## Rejection Types (DPO)

- `partial_factual_errors` - Some claims false; others correct
- `less_faithful_reconstruction` - Loses important nuance
- `unsupported_additions` - Plausible but unsupported claims
- `missing_information` - Incomplete; omits essentials
- `wrong_register` - Correct but wrong tone/formality/dialect
- `weak_organization` - Poor structure or explanation
- `poor_instruction_following` - Doesn't fully address question
- `wrong_formatting` - Violates formatting if required
- `verbosity` - Correct but unnecessarily lengthy

## Running the Pipeline

### Full Generation (All Chunks)
1. Set `SFT_CHUNK_LIMIT = 394` (or your chunk count)
2. Run sections 1–6 for SFT generation
3. Manually review section 7
4. Run section 8 to accept samples
5. Run section 9 for DPO generation
6. Save in section 10

### Quick Test (2–5 Chunks)
1. Set `SFT_CHUNK_LIMIT = 5`
2. Run sections 1–5 only
3. No DPO generation needed

### Smoke Test Only
- Run sections 1–3 only
- Tests prompt without generating data

## Prompt Specifications

### System Prompt
Located in `src/sft/prompts.py`
- Detailed instructions for question/answer/thinking generation
- Enforces Saudi Arabic, naturalness, and grounding
- Lists question types, difficulty levels, reasoning modes
- Specifies what NOT to do (meta-commentary, artificial language, etc.)

### Fallback Prompt
Simpler version used if primary prompt returns zero examples.
- Still enforces quality requirements
- Requires at least one example

### DPO Prompt
Generates plausible flawed alternatives.
- Ensures one meaningful flaw
- Prevents obviously absurd rejections
- Tracks rejection type

## Troubleshooting

### No API Key Error
```python
# Set in .env or provide interactively
export OPENROUTER_API_KEY="sk-..."
```

### Zero Examples Generated
- Check chunk quality in section 2
- Run smoke test (section 3) to debug
- Fallback prompt activates automatically

### Low Pass Rate
- Adjust acceptance threshold in section 8: `confidence >= 0.7`
- Review sample quality in section 7
- Check if chunks are too complex

### API Rate Limits
- Increase `PAUSE_BETWEEN_CALLS` in configuration
- Reduce `SFT_CHUNK_LIMIT` for testing

## Next Steps in Production

1. **Run Verification:** 
   ```bash
   python src/verification/verify_sft.py data/generated/sft/candidates.jsonl \
     --output data/generated/sft/accepted.jsonl
   ```

2. **Check DPO Quality:**
   ```bash
   python src/verification/check_dpo.py data/generated/dpo/candidates.jsonl
   ```

3. **Validate Release:**
   ```bash
   python src/verification/validate_release.py data/generated/sft/accepted.jsonl \
     --schema sft
   ```

## Architecture

```
src/sft/
├── __init__.py           # Package exports
├── schema.py             # Data models + enums
├── prompts.py            # System/user prompts
├── generator.py          # API calls + processing
└── validation.py         # Schema + quality checks

notebooks/
└── sft_dpo_pipeline_v2.ipynb  # Main workflow (11 sections)
```

All modules are typed, well-documented, and follow your specification exactly.

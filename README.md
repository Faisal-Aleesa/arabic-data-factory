# arabic-data-factory — Data Engineering

The Data Engineering stage of the **Arabic Post-Training Data Factory**. This stage turns
acquired source documents into clean, chunked, licence-tracked training data. It is fully
deterministic — **no LLM calls anywhere in the pipeline**.

The corpus currently in progress is a **Saudi regional dialect corpus** partitioned into
five regions: `najdi`, `southern`, `northern`, `eastern`, `western`.

---

## Repo structure

```
data/
  raw/<region>/        source extracts, as produced by the extraction tool
  interim/<region>/    per-document JSON: <doc_id>.json and <doc_id>_cleaned.json
  processed/           chunks.jsonl (the deliverable) + per-stage JSON reports
  eda/                 generated EDA report and token histogram
docs/
  license_manifest.csv       one row per source document — licence, rights status, provenance
  damaged_pages_review.csv   pages held back from chunking, for manual review
src/
  data_engineering/    this stage (see Pipeline below)
  reconstruction/      \
  verification/         |  placeholders owned by other team roles;
  dpo/                  |  empty by design, not this stage's scope
  deployment/          /
```

## Why `data/` looks empty after cloning

Two separate reasons, both intentional:

1. **Source content is gitignored pending licence resolution.** The current source's
   licence is `unverified_pending_review`, so nothing derived from its text is committed —
   not `data/raw/`, not `data/processed/chunks.jsonl`, not the EDA build that quotes
   example chunks. What *is* committed is metadata: stage reports, the manifest, and an
   excerpt-free EDA report.
2. **Region subfolders are created by the pipeline**, not committed. `data/*/*` is
   gitignored except for `.gitkeep`, so `najdi/`, `southern/` and friends appear the first
   time you run the pipeline. Nothing is broken if they're missing.

## Pipeline

Run in order, from the repo root:

```bash
python src/data_engineering/ingest_dialect_dictionary.py
python src/data_engineering/clean.py  --format dictionary
python src/data_engineering/dedup.py
python src/data_engineering/chunk.py  --format dictionary
python src/data_engineering/eda.py    --out data/eda/eda_report_saudi_dialect.md \
                                      --png data/eda/token_histogram_saudi_dialect.png \
                                      --no-examples
```

| stage | what it does |
|---|---|
| `ingest_dialect_dictionary.py` | acquisition: splits the source extract by verified page range, strips page furniture, repairs displaced diacritics, holds back damaged pages |
| `clean.py` | encoding repair, Arabic normalization, format-aware segmentation into units, per-doc review flags |
| `dedup.py` | exact (SHA-256) and near-duplicate (MinHash/LSH) detection at document level |
| `chunk.py` | packs whole units into 200–800 token chunks → `data/processed/chunks.jsonl` |
| `eda.py` | statistics, region distribution, review-flag summary → `data/eda/` |

### `--format` must match between `clean.py` and `chunk.py`

Format is **explicit — there is no auto-detection**. Choose one of `prose`, `dictionary`,
`verse`, and pass the *same* value to both stages. A mismatch is detected and refused
rather than silently producing garbage. The format determines what counts as an
indivisible unit: a paragraph, a glossary entry, or a stanza.

### Two flags worth knowing

- `chunk.py --text-variant` defaults to `original` (orthography preserved). The
  alef/ya-folded text exists **only** as a matching key for `dedup.py` and is never
  shipped — regional spelling variation is the signal this corpus exists to capture.
- `eda.py --no-examples` omits the one report section that reproduces source text
  verbatim. Use it for any report that will be committed or shared.

`dedup.py --self-test` verifies the duplicate detector against synthetic duplicates; a
run finding zero duplicates proves nothing on its own.

## Current corpus state

**337 chunks / 252,802 tokens**, 100% inside the 200–800 token target band.

| region | chunks | tokens |
|---|---:|---:|
| najdi | 130 | 97,703 |
| southern | 103 | 77,439 |
| northern | 49 | 37,574 |
| eastern | 30 | 22,007 |
| **western** | **25** | **18,079** |

**`western` is the sourcing priority** — 48% of the median regional token count. This is
inherent to the current source's coverage, not extraction loss, so it needs new material
rather than reprocessing.

Two review queues are open, neither auto-resolved:

- `docs/damaged_pages_review.csv` *(tracked)* — 41 pages held back from chunking where the
  source PDF's overlapping text runs extract as unreadable interleaved characters.
- `docs/residue_tokens_review.csv` *(gitignored)* — 5,619 short fragments that no
  dictionary-safe rejoin could confirm. Gitignored because its context column quotes
  readable source text.

## Before you commit

This project is deliberately careful about licence containment for sources whose rights
are not yet cleared.

- **Check `docs/license_manifest.csv`** before adding source material, and add a row for
  anything new — including `license`, `corpus_phase`, and provenance.
- **Do not commit anything containing substantial verbatim source text** until its licence
  status is confirmed. That includes chunk output, raw extracts, review logs that quote
  context, and EDA reports built without `--no-examples`.
- When in doubt, gitignore it and say so in the commit message. Metadata, statistics, and
  code are safe to commit; source text is not, until the manifest says otherwise.

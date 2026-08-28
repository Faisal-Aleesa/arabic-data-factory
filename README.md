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

## Adding a New Source

Everything below is what the first source actually required. Follow the order — the
licence and format decisions are cheap up front and expensive to reverse after processing.

### Step 1 — Licence first, before any processing

Add a row to `docs/license_manifest.csv` **before** you extract or run anything. Record
`license` (or `unverified_pending_review` if rights aren't cleared), `source_site` and
`license_note` with how you verified it and when, `region`, and `corpus_phase`.

If the status is unverified, gitignore everything derived from the text — see **Before you
commit** below. The current source is still `unverified_pending_review`, which is why
`data/` is mostly empty in the repo.

### Step 2 — Is it one region or many?

A single-region source is simple: put its extract in `data/raw/<region>/`.

A multi-region source needs page ranges, and **you must verify them against the source
itself — do not trust stated boundaries.** For the current source, every one of the five
stated start pages was one page late: each pointed at the section's first *content* page,
while the part-title page sat immediately before it. The stated end pages were wrong the
same way, each swallowing the next section's title page. The trailing sections (maps,
index) were off by one too.

The source's own table of contents was not reliable either — extracting it produced
row-misaligned page numbers because of its two-column layout. What worked was reading the
actual first and last page of each claimed range and confirming the heading text. Budget
an hour for this; it is the cheapest hour in the whole process.

### Step 3 — Determine `--format` before running `clean.py`

`--format` is explicit and must match between `clean.py` and `chunk.py`. Getting it wrong
does not raise an error — it produces plausible-looking, substantively broken output.

**Do not infer it from the title or genre.** The current source is a dictionary, but its
first section page reads as continuous prose; the glossary starts a few pages in. Sample
across the whole source, not the opening.

The method that worked: classify *every* page by the density of a candidate entry pattern
(lines matching `^(headword) :`), then look at the distribution. That showed 80–90%
glossary pages per region, with prose introductions and numbered feature lists recurring
at each sub-section. Conclusion: `dictionary` for all five regions, with the prose pages a
known minority the segmenter handles.

Choose `prose` for continuous text, `dictionary` for headword+definition entries, `verse`
for stanza-structured poetry. If a source genuinely mixes formats in large blocks, split it
into separate documents rather than forcing one format across all of it.

### Step 4 — Real text layer, or a scan needing OCR?

Check rather than assume. A page with no extractable text but embedded images is a scan:

```python
import pymupdf
doc = pymupdf.open("source.pdf")
for n in (10, 50, 100):                     # sample pages, don't trust page 1
    page = doc[n]
    print(n, len(page.get_text().strip()), len(page.get_images()))
# text length ~0 with images present  -> scanned, needs OCR
# substantial text length             -> real text layer, extract directly
```

For a real text layer, `pdftotext -layout -enc UTF-8` is what this project uses. It was
compared against `pdftotext` default / `-raw` / `-simple` / `-fixed` and against PyMuPDF on
the current source: `-layout` and the other order-preserving modes came out equivalent, and
PyMuPDF was worse (it fragments and reorders lines).

If OCR is needed, the usual recipe is PyMuPDF page-to-image plus `pytesseract` with
`lang="ara"`. **Neither `pytesseract` nor the `tesseract` binary is installed in this
project's environment**, so that is a setup task, not a ready path — verify output quality
on a sample before committing to it.

Either way, **OCR output must go through this project's `clean.py`**, not be cleaned
separately. The normalization decisions below are what keep sources comparable; a source
cleaned outside the pipeline silently opts out of them.

### Normalization decisions to revisit per source

These defaults were chosen for *this* source. Re-check each one rather than inheriting it.

**Diacritics — currently preserved** (`--strip-diacritics` off). In a dialect dictionary,
vocalization carries pronunciation information, which is much of the point of the source. A
different source type may carry no meaningful vocalization, or carry it inconsistently
enough to be noise. Decide deliberately.

**Tatweel — currently preserved** (`--strip-tatweel` off). This was a *specific finding for
this source*, not a general rule: it uses elongation expressively, and separately uses it
ornamentally inside complete words. In a typographically conventional source, tatweel is
just padding and stripping it is the right call. Sample before choosing.

**Alef/ya folding — dedup-only, always.** This one should hold for every source. Folding
`أ/إ/آ → ا` and `ى → ي` is lossy: it turns `نشأة` into `نشاة`. That is exactly what you want
for *matching*, because it collapses spelling variants of one word so duplicate detection
works — and exactly what you do not want in training text, where regional spelling
variation is the signal being collected. So `clean.py` writes both variants:
`paragraphs_original` (shipped) and `paragraphs` (folded, consumed only by `dedup.py`).
`chunk.py --text-variant` defaults to `original` and should stay that way.

### What the pipeline actually expects

`clean.py` does **not** read `data/raw/`. It globs `data/interim/*/*.json` (ignoring
`*_cleaned.json`). Only the ingest step reads `data/raw/`, so a new source needs something
that writes interim records in this shape:

| field | required | notes |
|---|---|---|
| `doc_id` | yes | unique; becomes the chunk id prefix |
| `title` | yes | used for front-matter detection |
| `author` | yes | copied onto every chunk |
| `source` | yes | provenance string, copied onto every chunk |
| `license` | yes | copied onto every chunk |
| `raw_text` | yes | the whole document as one UTF-8 string |
| `region` | in practice | omitting it does not crash, but every chunk groups under `null` |
| anything else | no | passed through onto the cleaned record |

Verified by dropping each field in turn and running the pipeline: missing `doc_id`, `title`
or `raw_text` fails in `clean.py`; missing `author`, `source` or `license` fails in
`chunk.py`.

### Reuse the ingest script, or write a new one?

**Recommendation: write a small new ingest script per source, reusing the helpers from
`ingest_dialect_dictionary.py`. Don't generalize it into one configurable script yet.**

Genuinely specific to the current source:

- `REGION_FIRST_PAGE` — the verified page numbers for this book
- `TITLE`, `AUTHOR`, `SOURCE`, `LICENSE` — this book's metadata
- the input filename pattern `dialect_dictionary_<region>.txt`
- `ABBREVIATIONS` — single letters that are legitimate in *this* dictionary (`ص` page, `ج`
  plural, `د` doctor) and would be fragments elsewhere

Reusable as-is:

- `page_lines` / `FOLIO_RE` — strips the printed folio; generic to any paginated book
- `damage_score` — flags pages whose extraction produced impossibly long pseudo-tokens (it
  caught 41 pages, 4.8%, in the current source)
- `repair_diacritics` — safe to run on any source: verified to be a **no-op on well-formed
  vocalized text**, firing only on the specific displacement artifact
- `residue_tokens` — collects short fragments for manual review
- the shape of `ingest_region` — split pages, drop furniture, exclude damaged, repair,
  reassemble, emit the record plus review rows

The reason not to generalize now: with one source, we cannot tell which parts actually
vary. Every source-specific item above is also a decision that required human verification
— page ranges, format, which single letters are legitimate. A generic ingest script would
need configuration for all of them anyway, and would make it easy to skip that verification
by accepting defaults. That is the failure mode worth avoiding. Revisit after the second or
third source, when the real pattern of variation is visible.

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

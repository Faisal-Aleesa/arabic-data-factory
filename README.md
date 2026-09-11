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
python src/data_engineering/clean.py  --corpus saudi_dialect --format dictionary
python src/data_engineering/dedup.py  --corpus saudi_dialect
python src/data_engineering/chunk.py  --corpus saudi_dialect --format dictionary
python src/data_engineering/eda.py    --out data/eda/eda_report_saudi_dialect.md \
                                      --png data/eda/token_histogram_saudi_dialect.png \
                                      --corpus-name "Saudi Regional Dialect Corpus" \
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

### `--corpus` scopes a run to one corpus

`data/interim/` can hold several corpora at once, and cleaning settings apply to every
document in a run. The dialect dictionary must KEEP tatweel; the classical lexicon must
have it stripped. A run spanning both would apply one corpus's settings to the other and
report success.

So `clean.py` and `chunk.py` **refuse to run** when more than one corpus is present and
no `--corpus` is given:

```
[clean] REFUSING to run: data/interim holds more than one corpus and no
    --corpus was given. Settings apply to every document in a run, so one
    corpus's settings would be applied to the other with no error raised.
    corpora found:
      classical_lexicon    1 document(s)
      saudi_dialect        5 document(s)
    Re-run scoped, e.g. --corpus classical_lexicon
```

With a single corpus present, an unscoped run behaves exactly as before.

`dedup.py` takes `--corpus` too but does **not** refuse without it — it applies no
per-corpus setting, and comparing across corpora is a legitimate question. Scope it when
the report is meant to describe one corpus.

The value comes from the `corpus` field each ingest script writes into its interim
records, using the same vocabulary as the manifest's `corpus_phase` column
(`saudi_dialect`, `classical_lexicon`). A new ingest script must set it.


### Two flags worth knowing

- `chunk.py --text-variant` defaults to `original` (orthography preserved). The
  alef/ya-folded text exists **only** as a matching key for `dedup.py` and is never
  shipped — regional spelling variation is the signal this corpus exists to capture.
- `eda.py --no-examples` omits the one report section that reproduces source text
  verbatim. Use it for any report that will be committed or shared.

`eda.py --corpus-name` names the report as a whole. Pass it for any corpus of more than
one document — without it a multi-document corpus falls back to a static heading rather
than being titled after whichever document happens to sort first.

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

### Step 5 — Give the source its own corpus name

Set a `corpus` value in the interim records your ingest script writes, and pass it as
`--corpus` to `clean.py`, `dedup.py` and `chunk.py`. Reuse the manifest's `corpus_phase`
vocabulary.

This is not bookkeeping. Cleaning settings are corpus-wide, so the moment a second corpus
exists in `data/interim/` an unscoped run would apply one corpus's `--format` and
tatweel/diacritic settings to the other — no error, plausible-looking output, wrong text.
`clean.py` and `chunk.py` now refuse rather than guess, but they can only do that if your
records carry the field.


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

### Install the pre-commit hook (one line, do this after cloning)

```bash
git config core.hooksPath .githooks
```

The hook lives in `.githooks/pre-commit` and is committed, so it travels with the repo —
but `core.hooksPath` is per-clone config, so **every clone needs that one line once**.
Without it the hook file is inert and the audit below is honour-system only.

Once installed, every `git commit` runs the audit against your staged changes:

| audit exit | hook behaviour |
|---|---|
| 0 — clean | commit proceeds |
| 1 — rights-pending text staged | **commit blocked** |
| 2 — no corpus available to check against | warns, commit proceeds |

Exit 2 is the normal state on a fresh clone: the dialect corpus is gitignored and the
held branch is local-only, so a machine that has never held the source text has nothing
to leak from it. Blocking every commit there would just get the hook turned off. On a
machine that does have the corpus, exit 1 bites.

`git commit --no-verify` bypasses the hook. Don't — the audit exists precisely because
source passages migrate into code during debugging, and code is not gitignored.

### The secrets scan (same hook, runs before the licence audit)

Added 2026-09-11 after a live API key was committed to `.env.example` and pushed to the
public remote. Every gate in the hook passed it: the licence audit scans for Arabic source
text and correctly reported the file as clean, because it was never a secrets scanner.
`.gitignore` covers `.env` but deliberately not `.env.example` — the example *is* the
template — so the one file guaranteed to be committed had no protection at all.

`src/verification/audit_staged_secrets.py` now runs on every staged file and refuses:

| what | fires on | pattern |
|---|---|---|
| OpenRouter key | any file | `sk-or-v1-` + 40+ alphanumerics |
| Anthropic key | any file | `sk-ant-` + 20+ |
| OpenAI-style key | any file | `sk-` + 20+ (only where the two above did not match, so a key is reported once under its real name) |
| AWS access key id | any file | `AKIA` + 16 uppercase/digits |
| any high-entropy value | `.env*` files only | a `KEY=value` that is 20+ chars, not placeholder-shaped, has no `/` `:` or spaces, and scores ≥ 3.5 bits/char |

| scanner exit | hook behaviour |
|---|---|
| 0 — nothing credential-shaped | commit proceeds |
| 1 — a key-shaped value is staged | **commit blocked** |
| 2 — the git index could not be read | **commit blocked** |

Note the last row: unlike the licence audit, the secrets scan's exit 2 **blocks**. The
licence audit's "no corpus" is the normal state of a fresh clone, where there is nothing
to leak. There is no innocent version of "could not see the staged files" — a scanner
that ran on nothing has no basis to call the commit safe.

Placeholders like `your_api_key_here`, `replace_with_your_openrouter_key`, `<your-key>`
or `CHANGEME` never fire; the self-test pins that. Neither do legitimate config values
such as model names or URLs. The `.env*` check was designed from measurement rather than
a guess: on this repo's own values, entropy alone *cannot* separate a placeholder from a
key — `google/gemini-2.5-flash-lite` scores higher than a raw hex key — so entropy is the
last gate, after a placeholder-word exemption and a no-`/`-no-`:` structural filter. The
scanner's docstring has the table.

Findings are reported as file, line, pattern and the value **redacted** to its first 8
characters — the scanner never reproduces the secret it refuses.

**If the scanner fires on a real key, revoke it at the provider first.** Replacing the
value in the file is the second step, not the first: a key that reached the public remote
is compromised whether or not a later commit removes it. Removing it does not un-leak it.

Run it by hand any time: `python src/verification/audit_staged_secrets.py` (staged
files) or `--files <paths>`; `--self-test` exercises every pattern, every placeholder
shape, and the scanner reading its own source.

### Why the licence audit is NOT in CI, and what that leaves open

**The licence audit cannot run in GitHub Actions, or on any remote runner.** This is a
structural consequence of the containment design, not a gap waiting to be filled.

`audit_staged_arabic.py` works by comparing staged text against the rights-pending
dialect corpus. That corpus is gitignored and has never been pushed, so a fresh clone —
which is exactly what a CI runner gets — does not have it. **Verified** on a clean clone:

```
[audit] REFUSING to pass: no rights-pending corpus could be loaded
    (saudi_dialect -> data/processed/chunks.jsonl).
```

That refusal is the tool working correctly. An audit with nothing to compare against
would report "clean" for any input, which is worse than not running it at all.

Making it run remotely would require one of three things, and **all three are ruled out**:

| option | why not |
|---|---|
| commit the dialect corpus | violates the hold this safeguard exists to enforce |
| put it in a GitHub secret or artifact store | that IS uploading rights-pending source text to a third party — the exact act being guarded against, minus the audit trail |
| self-hosted runner holding the corpus | technically works; needs a maintained machine, and the corpus would sit on it indefinitely |

**Do not treat this as a TODO.** Anyone proposing to "just add the audit to CI" is
proposing one of the three above. If that changes — for instance if the dialect source's
rights are cleared and the corpus becomes publishable — revisit this section then, not
before.

#### What this leaves open, stated plainly

**The licence audit is a PROCESS control, not a technical one.** The pre-commit hook
enforces it only for people who ran the one-line install, committing locally. It cannot
see:

- content pushed through GitHub's web uploader, which never touches a local hook
- anyone who runs `git commit --no-verify`
- anyone who simply has not installed the hook

**So anyone with write access can push unaudited Arabic content, and nothing in this
repository will stop them or notice.** That has already happened once: a web upload in
September 2026 added ten files and modified two, none of it screened. It audited clean
when checked afterwards by hand, but that was luck rather than control — and the same
upload silently deleted this README and `requirements.txt`, which is how the gap was
noticed at all.

**Branch protection plus human review is the only mitigation that exists for this gap.**
Requiring a pull request before merging to `main` makes every change — including web
uploads — visible in a diff someone has to look at, and gives a reviewer the chance to
pull the branch and run the audit locally. It is not automatic and it is not a guarantee:
a required approval on a small team under deadline can become a rubber stamp, in which
case it buys visibility, not prevention.

The rule that follows, for everyone with write access:

> **Run `python src/verification/audit_staged_arabic.py` locally before pushing anything
> containing Arabic text. No remote check will do it for you.**

#### What CI *can* usefully do

These need no corpus and run on a bare clone — **verified**: 98 cross-module references
resolve and 9/9 self-tests pass with no dialect data present.

```bash
python tools/check_symbols.py --cross-module src/verification   # catches silently-failed edits
python src/verification/<module>.py --self-test                 # all nine modules
```

Worth adding as a workflow when convenient. Be clear about what it buys: it catches broken
imports and regressions, **not** licence leaks.

### Required: run the staged-Arabic audit

**Before committing anything that touches dialect-derived content, stage your changes and
run this. It is not optional.**

```bash
python src/verification/audit_staged_arabic.py
```

Exit 0 means clean; exit 1 means verbatim rights-pending source text is staged and the
commit should not proceed as-is.

`.gitignore` already stops the obvious mistake — committing `chunks.jsonl` itself. This
catches the quiet one: source passages that migrate into **code** while you work.
Docstring examples, a regex tuned against a real line, a self-test fixture pasted from a
chunk you were debugging. Those files are not gitignored, so they go public.

This is not hypothetical. Auditing `src/verification/extract_facts.py` before its first
commit found ten verbatim dialect passages in its docstrings and self-test cases — a
definition, two section headings, a footnote and two real poet names — all pasted in
during debugging and all about to be pushed to a public repo. They were replaced with
invented placeholders (see the `الباب الثامن` convention: use examples that cannot be
mistaken for citations).

The script reads the **git index**, not the working tree, so it audits exactly what is
about to be committed. Every multi-word Arabic run is classified against two populations:

| verdict | meaning | action |
|---|---|---|
| `HELD_ONLY` | verbatim in a rights-pending source, in no public source | replace with an invented placeholder |
| `ALSO_IN_PUBLIC` | also present in a lawfully public corpus | review; usually a generic connective or shared classical material |
| *(allowlisted)* | already triaged in `docs/arabic_audit_allowlist.txt` | not reported |

Occurrence counts are printed to help triage: a phrase appearing hundreds of times is a
connective, one appearing once or twice is distinctive. Add a phrase to the allowlist only
after deciding it is genuinely not source content — never merely to silence the audit.

Single words are deliberately not reported; a lone place name, headword or root is not
meaningfully source text, and reporting them buried the real findings.

The audit refuses to pass (exit 2) if it cannot load a rights-pending corpus — the dialect
corpus is gitignored and local-only, and an audit with nothing to compare against would
report "clean" for any input. It falls back to reading the corpus from
`held/dialect-corpus-release` when the on-disk copy is missing.

`python src/verification/audit_staged_arabic.py --self-test` verifies the detector,
including a live regression that samples a real held phrase at run time and confirms it is
still caught.

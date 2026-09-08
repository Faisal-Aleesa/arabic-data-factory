# Dataset card — TEMPLATE / OUTLINE

**⚠️ This is a skeleton, not a release artifact.** Sections and guidance only. Everything
marked `{{PLACEHOLDER}}` needs a real number or decision before publication. Facts that are
already measured are filled in and marked **[measured]**; do not re-derive them, but DO
re-verify them at release time, since the corpus may have grown.

**بطاقة مجموعة البيانات — قالب، وليست نسخة نهائية.** كل ما بين `{{ }}` ينتظر رقمًا أو
قرارًا حقيقيًا قبل النشر.

**Before publishing, delete this banner and every `{{PLACEHOLDER}}` must be resolved.**

---

## YAML front matter

Hugging Face reads a YAML block at the very top of `README.md`. Fill and move above the
title:

```yaml
language: [ar]
license: {{LICENSE_ID}}          # NOT a single value - see the licensing section
task_categories: [text-generation]
size_categories: [{{SIZE_BUCKET}}]
configs: {{SFT_AND_DPO_CONFIGS}}
tags: [arabic, saudi-dialect, classical-arabic, lexicography, sft, dpo]
```

> ⚠️ `license:` is a single field, and this dataset does **not** have a single licence.
> See **Sources and licensing** below and resolve deliberately — do not pick whichever
> value makes the form validate.

## Dataset summary

- One paragraph: what this is, what it was built from, what it is for.
- Name the two corpora and that they are different in kind (regional dialect lexicon vs
  classical rhetoric lexicon), not two samples of one thing.
- State plainly that the corpus is small: **[measured]** 731 chunks / 538,302 tokens
  across 6 source documents.

## Sources and licensing

> **Do not summarise the licence in prose alone.** Point at
> [`docs/license_manifest.csv`](license_manifest.csv), which carries per-document
> `license`, `license_note`, `source_site`, `url`, page ranges, extraction method and
> `date_checked`, and is the authoritative record.

| source | documents | licence | attribution |
|---|---|---|---|
| أساس البلاغة (Arabic Wikisource) | `asas_albalagha` | **[measured]** CC BY-SA 4.0 | **[measured]** `https://ar.wikisource.org/wiki/%D8%A3%D8%B3%D8%A7%D8%B3_%D8%A7%D9%84%D8%A8%D9%84%D8%A7%D8%BA%D8%A9` |
| معجم اللهجات المحكية (supplied PDF) | 5 × `dialect_dict_*` | **[measured]** `permission_granted` | `{{PERMISSION_HOLDER_FULL_NAME}}`, `{{DATE}}`, `{{MEDIUM}}` |

**Blocking before publication:**
- `{{PERMISSION_HOLDER_FULL_NAME}}` — the manifest currently records a first name only.
  Needed: full name, relationship to the rights holder, date, medium. See the hard rule in
  the session handoff.
- Confirm the permission covers **public publication of derived records**, not only
  training use. These are different, and the dataset card is a publication.
- CC BY-SA is **share-alike**: state what that obliges downstream users to do, and confirm
  the chosen dataset licence is compatible with it for the classical half.

## Corpus statistics

**[measured]** — re-verify at release:

| document | chunks | tokens |
|---|---:|---:|
| `asas_albalagha` | 394 | 285,500 |
| `dialect_dict_najdi` | 130 | 97,703 |
| `dialect_dict_southern` | 103 | 77,439 |
| `dialect_dict_northern` | 49 | 37,574 |
| `dialect_dict_eastern` | 30 | 22,007 |
| `dialect_dict_western` | 25 | 18,079 |
| **total** | **731** | **538,302** |

Derived-record counts, which do not exist yet:

| | SFT | DPO |
|---|---|---|
| records / pairs | `{{N_SFT}}` | `{{N_DPO}}` |
| train / val / test | `{{SPLIT_COUNTS}}` | `{{SPLIT_COUNTS}}` |
| per source document | `{{PER_DOC_TABLE}}` | `{{PER_DOC_TABLE}}` |
| per `format_type` | `{{FORMAT_TABLE}}` | — |
| per `rejection_type` | — | `{{REJECTION_TYPE_TABLE}}` |

> Report `western` explicitly rather than letting it disappear into a total. **[measured]**
> it is the under-resourced region at 48% of the median regional token count, and users
> training on regional balance need to know that before they discover it.

## Record schema

Show one real example per config, plus the field table.

**SFT:** `instruction`, `response`, `source_chunk_id`, `source_region`, `format_type`,
`model_version`

**DPO:** `prompt`, `chosen`, `rejected`, `source_chunk_id`, `source_region`,
`rejection_type`, `model_version`

Document for each field: type, whether it is always present, and its vocabulary where
closed. In particular:
- `source_region` — the five dialect regions plus `classical`
- `format_type` — **[measured]** the corpus is 100% `dictionary_entry`; state whether
  released records inherit that or describe the response's own shape (see the policy in
  `src/deployment/ACCEPTANCE_CONTRACT.md` — it is the response's intended shape)
- `rejection_type` — the nine charter weakness types
- `source_chunk_id` — `<doc_id>_c<NNNN>`; note that the parent document is the split unit

## Splits

Summarise [`docs/SPLIT_POLICY.md`](SPLIT_POLICY.md) and link it. Must state:
- split is by **parent document**, never by chunk, so no document's records cross
  partitions
- **[measured]** there are only 6 splittable units, very unevenly sized
- therefore `{{SPLIT_DECISION}}` — and what val/test actually measure under it. If a
  dialect region is held out, val/test measure **cross-region generalisation**, which is a
  different claim from in-distribution accuracy and must be labelled as such.

## Generation and verification methodology

Brief description of how records were produced (`{{GENERATION_METHOD}}`,
`{{MODELS_USED}}`), then the verification stack. Link the modules; do not restate their
internals.

| check | what it establishes |
|---|---|
| `extract_facts.py` | ground-truth fact table per chunk; offsets re-verified against source text |
| `check_facts.py` | response facts vs the table — SUPPORTED / PARTIAL / UNSUPPORTED / CONTRADICTED |
| `check_similarity.py` | semantic grounding against the source chunk, reported as a percentile |
| `check_format.py` | structural conformance to the declared `format_type` |
| `verify_sft.py` | consolidated verdict + `acceptance_decision()` gate |
| `check_dpo.py` | pair triage, `rejection_type` corroboration, chosen/rejected distinctness floor |
| `diversity.py` | instruction phrasing variety, region and format spread |
| `validate_release.py` | last-mile JSON, encoding and orthography validation |

Report the **outcome distribution** on the real released set:
`{{ACCEPT_HOLD_REJECT_COUNTS}}`, `{{DPO_TRIAGE_COUNTS}}`, `{{JUDGE_AGREEMENT_RATE}}`.

> State honestly what proportion was human-reviewed vs automated-only. `HOLD_FOR_REVIEW`
> is expected to be common by design; publishing without saying how those were resolved
> would misrepresent the process.

## Known limitations

> These are already documented in the modules. Carry them across rather than writing
> softer versions — a limitation a user discovers themselves costs more trust than one
> disclosed up front.

**Coverage of the verification stack**
- **[measured]** 2 of the 9 DPO weakness types (`wrong_register`, `weak_organization`) are
  invisible to every automated check; 4 of 9 can be auto-confirmed. Judge/human review
  carries the rest.
- `verbosity` is reliably detected but can never be auto-confirmed — a padded answer keeps
  every fact, so verdict and similarity tie.

**Fact checking**
- Lexical items are only detected when the response **marks** them (parentheses/quotes). A
  hallucinated lemma in plain prose can pass.
- The checker inherits extractor recall: a fact the extractor missed reads as
  `UNSUPPORTED`, so those counts are not a hallucination rate. Read alongside
  `docs/citation_review_*.csv`.
- Region claims rest on `cross_dialect_reference` facts, whose extraction recall is poor.

**Semantic similarity**
- Drift and hallucination are **not separable** — both read as ungrounded.
- Partly tracks lexical overlap, not pure semantics: the same paraphrase probe scored
  pct 96 on classical and pct 20 on dialect. Low scores are evidence of drift, not proof.
- **Gameable:** appending a formulaic citation clause moved a hallucinated response from
  pct 31.9 to 90.8. Never a sole gate.
- Scores are model-dependent; record the model id.

**Format**
- **[measured]** only `dictionary_entry` is validated against real corpus content; the
  other five format types rest on synthetic cases.
- `prose` and `narrative_paragraph` are accepted interchangeably.

**Diversity**
- Phrasing variety is **not** task variety — differently-worded instructions can all ask
  the same thing.
- A region-imbalance warning may reflect the corpus's own sourcing gap rather than a
  sampling defect.

**Release validation**
- Validates **form, not content**; short responses are not orthography-checked; tatweel is
  deliberately unchecked because it is a per-source decision here.

**Thresholds**
- Every band in the stack (similarity, distinctness, diversity) is **provisional**, derived
  from hand-built probes, and should be recalibrated against `{{REAL_DATA}}`.

**Corpus**
- Small (731 chunks), 6 documents, uneven regional coverage.
- **[measured]** 41 damaged source pages excluded; 5,619 residue fragments logged for
  review — both tracked in `docs/`.

## Intended use

- **Intended:** `{{INTENDED_USES}}` — e.g. instruction tuning for Arabic lexicographic
  QA, dialect-aware generation research.
- **Out of scope:** state plainly. Candidates: authoritative lexicographic reference,
  dialect identification for real-world decisions, anything safety-critical.
- **Not a language authority.** Records are model-generated from two specific dictionaries
  and reflect those sources' coverage, era and editorial choices — including a print
  edition dated 1434H and a classical text.
- Note the dialect/classical asymmetry: **[measured]** over half the chunks are classical,
  so a model trained on this is not a balanced Saudi-dialect model.

## Ethical considerations and bias

- Regional imbalance is a representational issue, not only a statistical one: `western`
  and `eastern` speakers are the least represented.
- Dictionary sources carry the editorial and social assumptions of their period.
- `{{PII_STATEMENT}}` — confirm whether the sources contain personal names of living
  people beyond cited authors.

## Citation and provenance

- `{{CITATION_BLOCK}}` for the dataset.
- Cite both source works. Attribution for `asas_albalagha` is a **CC BY-SA requirement**,
  not a courtesy.
- Link the repository and the commit or release tag the data was built from.

## Maintenance

- `{{CONTACT}}`, `{{VERSIONING_POLICY}}`, how corrections are reported.
- Note that `western` sourcing is an open gap and future versions may rebalance.

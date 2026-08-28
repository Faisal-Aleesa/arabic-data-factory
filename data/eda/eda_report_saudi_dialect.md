# EDA Report - Saudi Regional Dialect Corpus

- **Source corpus:** provided directly by the team, not from a public repo | **License:** unverified_pending_review
- **Documents in:** 5
- **Source format:** `dictionary` (explicit per batch; clean.py and chunk.py must agree)
- **Chunks out:** 337
- **Total tokens:** 252,802 (*whitespace_word_count* approximation)
- **Pipeline:** `clean.py` -> `dedup.py` -> `chunk.py` -> `eda.py` (fully deterministic, no LLM calls)
- **Text variant chunked:** `original` - orthography preserved; normalization was used for matching/dedup only.

## 1. Token count distribution

`token_count` is a **word-based approximation**: whitespace-delimited words (`len(text.split())`). No subword tokenizer is applied at this stage, so the pipeline stays deterministic and model-agnostic. For Arabic, a SentencePiece/BPE tokenizer typically yields ~1.5-2.5 subword tokens per word.

| statistic | tokens |
| --- | ---: |
| min | 170 |
| p25 | 751 |
| median (p50) | 782 |
| p75 | 793 |
| p90 | 798 |
| max | 800 |
| mean | 750.2 |

**335 / 337 chunks (99.4%) fall inside the 200-800 token target band.**

```
tokens/chunk        | histogram                                     count
   170-   222 | #                                                 3
   222-   275 |                                                   1
   275-   328 |                                                   0
   328-   380 |                                                   1
   380-   432 |                                                   0
   432-   485 | #                                                 5
   485-   538 |                                                   1
   538-   590 | #                                                 7
   590-   642 | ##                                               11
   642-   695 | #                                                 5
   695-   748 | ########                                         43
   748-   800 | ##############################################  260
```

![Token distribution](token_histogram_saudi_dialect.png)

## 2. Region distribution

| region | docs | chunks | share | tokens | median tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| najdi | 1 | 130 | 38.6% | 97,703 | 772 |
| southern | 1 | 103 | 30.6% | 77,439 | 783 |
| northern | 1 | 49 | 14.5% | 37,574 | 787 |
| eastern | 1 | 30 | 8.9% | 22,007 | 793 |
| western | 1 | 25 | 7.4% | 18,079 | 782 |
| **total** | 5 | **337** | 100.0% | **252,802** | 782 |

### ⚠ Under-resourced regions - priority for additional sourcing

| region | tokens | vs. median region | vs. largest region |
| --- | ---: | ---: | ---: |
| western | 18,079 | 48% | 19% |

**western** falls below 50% of the median regional token count. Additional sourcing here would do more for regional balance than more volume anywhere else in the corpus.

### Chunks per document

| doc_id | region | unit type | units | chunks | tokens |
| --- | --- | --- | ---: | ---: | ---: |
| dialect_dict_eastern | eastern | entry_headword | 872 | 30 | 22,007 |
| dialect_dict_najdi | najdi | entry_headword | 2,159 | 130 | 97,703 |
| dialect_dict_northern | northern | entry_headword | 1,192 | 49 | 37,574 |
| dialect_dict_southern | southern | entry_headword | 2,830 | 103 | 77,439 |
| dialect_dict_western | western | entry_headword | 791 | 25 | 18,079 |

### format_type distribution

| format_type | chunks | share |
| --- | ---: | ---: |
| dictionary_entry | 337 | 100.0% |

## 3. Cleaning stage and items flagged for review

- Characters in: **1,582,113** -> retained: **1,530,737** (**3.25%** removed overall)
- Flag threshold: a document is flagged when cleaning removes more than **30%** of its characters
- Diacritic stripping: **OFF** (default off - two children's books in this batch are fully vocalized)

Largest character deltas:

| doc_id | region | chars in | chars retained | % removed | flags |
| --- | --- | ---: | ---: | ---: | --- |
| dialect_dict_northern | northern | 237,028 | 226,597 | 4.40% | - |
| dialect_dict_najdi | najdi | 611,720 | 590,744 | 3.43% | - |
| dialect_dict_southern | southern | 481,462 | 467,777 | 2.84% | - |
| dialect_dict_eastern | eastern | 138,178 | 134,508 | 2.66% | - |
| dialect_dict_western | western | 113,725 | 111,111 | 2.30% | - |

### Documents flagged during cleaning: **0**

None.

### Chunks flagged

| flag | chunks | meaning |
| --- | ---: | --- |
| below_target_min | 2 | below 200 tokens - end-of-document tail, kept |

| chunk_id | region | tokens | units | flags |
| --- | --- | ---: | ---: | --- |
| dialect_dict_eastern_c0029 | eastern | 170 | 5 | below_target_min |
| dialect_dict_western_c0016 | western | 191 | 14 | below_target_min |

## 4. Deduplication

| metric | value |
| --- | ---: |
| documents scanned | 5 |
| exact duplicate groups (SHA-256) | 0 |
| documents removed as exact duplicates | 0 |
| MinHash/LSH candidate pairs | 0 |
| near-duplicate pairs confirmed | 0 |
| documents kept | 5 |

- Exact: `sha256 of whitespace-collapsed cleaned_text`
- Near: `MinHash+LSH, word 5-shingles, num_perm=128`, Jaccard threshold **0.8**, LSH candidates re-checked against the true Jaccard of the shingle sets to drop false positives
- Policy: `flag only, never auto-remove`

**Zero duplicates found** - expected for distinct documents from a single source. Because a clean run proves nothing about the detector itself, `dedup.py --self-test` injects an exact copy and a ~0.85-Jaccard perturbed copy of a real document and asserts both are caught while an unrelated document is not. It passes.

## 5. Example chunks

_Omitted from this build (`eda.py --no-examples`). This section is the only part of the report that reproduces source text verbatim, and this corpus carries `license: unverified_pending_review`. Regenerate without the flag for a local copy with excerpts._

---

Generated by `src/data_engineering/eda.py`.

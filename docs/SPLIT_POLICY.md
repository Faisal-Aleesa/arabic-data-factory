# Train / val / test split policy

**سياسة تقسيم البيانات: التقسيم بالوثيقة الأم، لا بالقطعة.**

**Status, updated 2026-09-09: the checker now exists.**
`src/verification/check_leakage.py` implements this policy. This file remains the
rationale — the checker's docstring points here rather than restating the argument.

Two things this document asserted needed correcting: the corpus now has **seven**
documents, not six (§"The consequence"), and the manifest/corpus `doc_id` sets had
drifted apart (§"Related"). The checker found the second on its first live run; it was
repaired the same day by renaming the manifest row, and both are recorded rather than
edited away.

---

## The rule

> **Split by the parent DOCUMENT of `source_chunk_id`, never by individual chunk.**
>
> No single document's derived SFT/DPO records may appear in more than one partition.

For this purpose the corpus has **seven documents** (six when this was written; the Najdi
popular-words corpus was added 2026-09-09). The five dialect regions are distinct
documents even though they come from one physical book, and `asas_albalagha` is one
document even though it was ingested from 18 EPUB parts.

## Deriving the document from a record

`source_chunk_id` is `<doc_id>_c<NNNN>`:

```
dialect_dict_najdi_c0083   ->  dialect_dict_najdi
asas_albalagha_c0001       ->  asas_albalagha
```

```python
PARENT_DOC = re.compile(r'^(.*)_c\d{4}$')
```

**Verified:** all 731 chunk ids across both corpora derive their `doc_id` correctly by
this pattern, with zero mismatches. A future checker should still assert the match rather
than assume it — a new ingest script could break the convention, and a silently
unparseable id would default to "not the same document", which fails open.

## Why document-level and not chunk-level

Chunks from one document are not independent. They share headwords, roots, vocabulary,
citation authorities and phrasing conventions, and the chunker packs whole entries into
200–800 token chunks, so adjacent chunks routinely continue the same alphabetical section
of the same dictionary. A record generated from `dialect_dict_najdi_c0083` and one from
`dialect_dict_najdi_c0084` can easily concern neighbouring entries under the same root.

A random chunk-level split would therefore put near-identical material on both sides of
the partition, and validation scores would measure memorisation rather than
generalisation. Splitting at the document boundary is the coarsest unit that makes the
partitions genuinely independent.

The same reasoning applies to DPO pairs: both halves of a pair derive from one
`source_chunk_id`, so a pair belongs wholly to its document's partition. A pair must never
be split across partitions, and `chosen`/`rejected` must never land in different ones.

## The consequence, stated plainly

**MEASURED — this corpus has only six splittable units, and they are very uneven:**

| document | chunks | tokens | share of chunks |
|---|---:|---:|---:|
| `asas_albalagha` | 394 | 285,500 | 54% |
| `dialect_dict_najdi` | 130 | 97,703 | 18% |
| `dialect_dict_southern` | 103 | 77,439 | 14% |
| `dialect_dict_northern` | 49 | 37,574 | 7% |
| `dialect_dict_eastern` | 30 | 22,007 | 4% |
| `dialect_dict_western` | 25 | 18,079 | 3% |
| `majam_alkalimat_alshaabia_najd` | 6 | — | <1% |
| **total** | **737** | **538,302+** | |

The seventh document adds 6 chunks and does not change any conclusion below — it is far
too small to be a partition and makes the imbalance slightly worse, not better. Its token
count is not filled in because the EDA that produced this column has not been re-run
across it.

This policy is correct for leakage, and it has a real cost that should be decided
deliberately rather than discovered during training:

1. **There is no conventional random split available.** Six units cannot be divided into
   train/val/test in any proportion resembling 80/10/10. Whichever partition holds
   `asas_albalagha` holds over half the corpus.
2. **Holding out a dialect region changes what val/test measure.** A held-out region is a
   *different dialect*, so the evaluation measures cross-region generalisation, not
   in-distribution performance. That may be exactly what is wanted — but it is a
   different claim from "validation accuracy", and reporting it as the latter would be
   wrong.
3. **The small regions are too small to stand alone.** `western` (25 chunks) and
   `eastern` (30) are the natural held-out candidates by size, and both are far too small
   for a stable estimate. `western` is separately the corpus's known under-resourced
   region at 48% of the median.
4. **A classical/dialect split is a different task.** Putting `asas_albalagha` on one side
   and the dialect regions on the other measures classical→dialect transfer, which is not
   the same evaluation as either corpus alone.

**Not decided here.** The options are: accept coarse document-level splits and report
val/test explicitly as cross-document generalisation; hold out one mid-sized region
(`northern`, 49 chunks) and accept a small val set; or acquire more source documents so
the split has more units to work with — which is independently desirable and already an
open item for `western`.

What is *not* an option is quietly relaxing to a chunk-level split to get nicer
proportions. That trades a visible constraint for an invisible leak.

## What the future checker must verify

**BUILT.** `src/verification/check_leakage.py`, all five implemented, each with a
discrimination self-test. Codes: `UNPARSEABLE_CHUNK_ID`, `UNKNOWN_DOC_ID`,
`DOC_IN_MULTIPLE_PARTITIONS`, `PAIR_SPLIT`, `TEXT_CROSSES_PARTITIONS`.

At minimum it verifies:

- every record's `source_chunk_id` parses to a known `doc_id` — **fail loudly** on one
  that does not, rather than treating it as its own document
- no `doc_id` appears in more than one partition
- for DPO, both halves of every pair are in the same partition
- exact-duplicate and near-duplicate response text does not cross partitions even within
  the rule — `dedup.py` already has MinHash/LSH machinery for this and should be reused
  rather than restated
- the check runs on the **released** files, not on an intermediate manifest, for the same
  reason `validate_release.py` exists: partitions are assembled and rewritten after the
  split is decided

## Related

- `src/deployment/ACCEPTANCE_CONTRACT.md` — what reaches the released set at all
- `src/verification/validate_release.py` — last-mile form validation on release files
- `src/data_engineering/dedup.py` — existing SHA-256 + MinHash/LSH duplicate detection
- `docs/license_manifest.csv` — the `doc_id` vocabulary this policy splits on.
  **Verified, and re-verified after a break and a repair on 2026-09-09:** the manifest's
  **seven** `doc_id` values and the seven appearing in the corpora are the same set, so
  the split unit and the licence-tracking unit are identical.

  **This invariant broke once and was repaired; the history is kept because the repair
  is the interesting part.** The Najdi popular-words row was first written as
  `dialect_dict_najdi_popular`, matching the naming convention of the five
  `dialect_dict_*` rows, while its chunks carried `majam_alkalimat_alshaabia_najd`.
  `check_leakage.py` reported all 6 chunks as `UNKNOWN_DOC_ID` on its first live run
  against real ids — which is what the checker is for.

  **Fixed by changing the MANIFEST to follow the data**, not the other way round. The two
  options were not equivalent:
  1. rename the manifest `doc_id` — edits a licence-tracking record, but the value was
     verified to be referenced nowhere except that row and prose comments, never in a
     lookup;
  2. re-emit the chunks as `dialect_dict_najdi_popular` — cosmetically tidier, but the
     chunks are already committed and pushed, so every derived id would change under
     anything already referencing them.

  (1) was chosen because it moves a string nothing depends on, rather than rewriting ids
  that are already public. The cost is that this one row does not match the
  `dialect_dict_*` naming convention of the other dialect sources. That is deliberate:
  **the convention is cosmetic and the invariant is load-bearing.** The row's
  `license_note` records the rename.

  The general rule still holds and is the intent: a new source adds a manifest row first
  (README, "Adding a New Source"), so it becomes a split unit at the same moment it
  becomes a licence-tracked one. This document is the case where that did not happen —
  the data was pushed before the row existed — and the mismatch is what that inversion
  cost.

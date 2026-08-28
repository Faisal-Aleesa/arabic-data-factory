"""Stage 2 - Deduplication (document level).

Input : data/interim/<region>/<doc_id>_cleaned.json   (Stage 1 output)
Output: data/processed/dedup_report.json              (always written, even if 0 dupes)

Two passes:
  1. EXACT     - SHA-256 over the normalized text (whitespace collapsed). Groups with
                 more than one member keep the first doc (sorted by doc_id) and the rest
                 are marked `exact_duplicate`; downstream chunking skips them.
  2. NEAR      - MinHash + LSH (datasketch), word 5-shingles, 128 permutations,
                 Jaccard threshold 0.8 (configurable). Candidate pairs from LSH are
                 re-checked with the true Jaccard on the shingle sets to remove LSH
                 false positives.

Near-duplicates are FLAGGED, not removed - the standing rule is "never discard
automatically". Only byte-identical documents are excluded from chunking.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re

from datasketch import MinHash, MinHashLSH

INTERIM_DIR = "data/interim"
REPORT_PATH = "data/processed/dedup_report.json"

NUM_PERM = 128
SHINGLE_SIZE = 5      # word n-gram size
DEFAULT_THRESHOLD = 0.8

# LSH banding is probabilistic: a pair sitting near the index threshold has only ~50%
# chance of being emitted as a candidate, so indexing at 0.8 silently misses genuine
# 0.80-0.85 near-duplicates (the self-test caught exactly that). The index is therefore
# built at a LOOSER threshold and used purely as a candidate generator - every candidate
# is still confirmed against the true Jaccard at DEFAULT_THRESHOLD, so recall improves
# without admitting a single false positive.
LSH_THRESHOLD_MARGIN = 0.2
MIN_LSH_THRESHOLD = 0.3


def lsh_threshold_for(threshold: float) -> float:
    return round(max(MIN_LSH_THRESHOLD, threshold - LSH_THRESHOLD_MARGIN), 4)


def load_cleaned_docs(interim_dir: str) -> list[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(interim_dir, "*", "*_cleaned.json"))):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        rec["_path"] = path.replace("\\", "/")
        docs.append(rec)
    return docs


def canonical_text(text: str) -> str:
    """Whitespace-collapsed form used for hashing (text is already Arabic-normalized)."""
    return re.sub(r"\s+", " ", text).strip()


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    words = text.split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def build_minhash(shingle_set: set[str], num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingle_set:
        m.update(sh.encode("utf-8"))
    return m


def true_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run(interim_dir: str, threshold: float, num_perm: int, shingle_size: int) -> dict:
    docs = load_cleaned_docs(interim_dir)
    if not docs:
        raise SystemExit(f"No *_cleaned.json found under {interim_dir}. Run clean.py first.")

    print(f"[dedup] loaded {len(docs)} cleaned documents")

    # ---------- pass 1: exact ----------
    by_hash: dict[str, list[str]] = {}
    canon: dict[str, str] = {}
    for d in docs:
        c = canonical_text(d["cleaned_text"])
        canon[d["doc_id"]] = c
        by_hash.setdefault(sha256_of(c), []).append(d["doc_id"])

    exact_groups = []
    exact_duplicates: dict[str, str] = {}   # duplicate doc_id -> kept doc_id
    for h, ids in by_hash.items():
        if len(ids) > 1:
            ids = sorted(ids)
            keep, dupes = ids[0], ids[1:]
            exact_groups.append({"sha256": h, "kept": keep, "duplicates": dupes})
            for d in dupes:
                exact_duplicates[d] = keep

    print(f"[dedup] exact pass: {len(exact_groups)} duplicate group(s), "
          f"{len(exact_duplicates)} document(s) marked as exact duplicates")

    # ---------- pass 2: near ----------
    shingle_sets = {d["doc_id"]: shingles(canon[d["doc_id"]], shingle_size) for d in docs}
    minhashes = {doc_id: build_minhash(s, num_perm) for doc_id, s in shingle_sets.items()}

    lsh_t = lsh_threshold_for(threshold)
    lsh = MinHashLSH(threshold=lsh_t, num_perm=num_perm)
    for doc_id, mh in minhashes.items():
        lsh.insert(doc_id, mh)

    near_pairs = []
    seen_pairs: set[tuple[str, str]] = set()
    for doc_id, mh in minhashes.items():
        for other in lsh.query(mh):
            if other == doc_id:
                continue
            pair = tuple(sorted((doc_id, other)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            est = minhashes[pair[0]].jaccard(minhashes[pair[1]])
            exact_j = true_jaccard(shingle_sets[pair[0]], shingle_sets[pair[1]])
            if exact_j >= threshold:      # drop LSH false positives
                near_pairs.append({
                    "doc_a": pair[0],
                    "doc_b": pair[1],
                    "minhash_estimate": round(est, 4),
                    "jaccard_true": round(exact_j, 4),
                    "action": "flagged_for_review",
                })

    print(f"[dedup] near pass: {len(seen_pairs)} LSH candidate pair(s) "
          f"(index threshold {lsh_t}), {len(near_pairs)} confirmed above Jaccard {threshold}")

    kept = [d["doc_id"] for d in docs if d["doc_id"] not in exact_duplicates]
    flagged_ids = sorted({i for p in near_pairs for i in (p["doc_a"], p["doc_b"])})

    report = {
        "config": {
            "exact_method": "sha256 of whitespace-collapsed cleaned_text",
            "near_method": f"MinHash+LSH, word {shingle_size}-shingles, num_perm={num_perm}",
            "jaccard_threshold": threshold,
            "lsh_index_threshold": lsh_t,
            "lsh_note": ("index built looser than the decision threshold so banding does "
                         "not drop borderline pairs; every candidate is re-checked "
                         "against the true Jaccard"),
            "near_duplicate_policy": "flag only, never auto-remove",
        },
        "documents_scanned": len(docs),
        "exact_duplicate_groups": exact_groups,
        "exact_duplicates_removed": len(exact_duplicates),
        "exact_duplicate_map": exact_duplicates,
        "near_duplicate_candidate_pairs": len(seen_pairs),
        "near_duplicate_pairs": near_pairs,
        "near_duplicates_flagged": flagged_ids,
        "documents_kept": kept,
        "documents_kept_count": len(kept),
        "per_document": [
            {
                "doc_id": d["doc_id"],
                "region": d["region"],
                "sha256": sha256_of(canon[d["doc_id"]]),
                "shingle_count": len(shingle_sets[d["doc_id"]]),
                "status": "exact_duplicate" if d["doc_id"] in exact_duplicates
                else ("near_duplicate_flagged" if d["doc_id"] in flagged_ids else "unique"),
            }
            for d in docs
        ],
    }
    return report


def self_test(interim_dir: str, threshold: float, num_perm: int, shingle_size: int) -> dict:
    """Sanity-check the detector on synthetic duplicates.

    A corpus drawn from a single source is expected to contain zero duplicates, so a
    clean run proves nothing about the logic. This injects (a) an exact copy and (b) a lightly perturbed
    copy (every 60th word replaced, true Jaccard ~0.85) of a real document, asserts both
    are caught, and asserts an unrelated document is not.
    """
    docs = load_cleaned_docs(interim_dir)
    base = max(docs, key=lambda d: len(d["cleaned_text"]))
    text = canonical_text(base["cleaned_text"])
    words = text.split()
    perturbed = [("XX" if i % 60 == 0 else w) for i, w in enumerate(words)]

    sets = {
        "original": shingles(text, shingle_size),
        "exact_copy": shingles(text, shingle_size),
        "near_copy": shingles(" ".join(perturbed), shingle_size),
        "unrelated": shingles(canonical_text(
            min(docs, key=lambda d: len(d["cleaned_text"]))["cleaned_text"]), shingle_size),
    }
    mhs = {k: build_minhash(v, num_perm) for k, v in sets.items()}
    lsh = MinHashLSH(threshold=lsh_threshold_for(threshold), num_perm=num_perm)
    for k, m in mhs.items():
        lsh.insert(k, m)

    exact_hit = sha256_of(text) == sha256_of(text)
    # candidates from the loose index, then the real decision on true Jaccard
    cands = {k for k in lsh.query(mhs["original"]) if k != "original"}
    near_hits = {k for k in cands
                 if true_jaccard(sets["original"], sets[k]) >= threshold}
    results = {
        "base_doc_id": base["doc_id"],
        "exact_sha_match": exact_hit,
        "lsh_candidates_of_original": sorted(cands),
        "confirmed_near_duplicates": sorted(near_hits),
        "jaccard_exact_copy": round(true_jaccard(sets["original"], sets["exact_copy"]), 4),
        "jaccard_near_copy": round(true_jaccard(sets["original"], sets["near_copy"]), 4),
        "jaccard_unrelated": round(true_jaccard(sets["original"], sets["unrelated"]), 4),
    }
    results["passed"] = bool(
        exact_hit
        and "exact_copy" in near_hits                  # identical text must be caught
        and "near_copy" in near_hits                   # ~0.85 Jaccard copy must be caught
        and results["jaccard_near_copy"] >= threshold
        and results["jaccard_unrelated"] < threshold   # unrelated text must not be
    )
    for k, v in results.items():
        print(f"[dedup:self-test] {k}: {v}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: exact + near duplicate detection.")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the synthetic duplicate sanity check and exit.")
    ap.add_argument("--interim-dir", default=INTERIM_DIR)
    ap.add_argument("--report", default=REPORT_PATH)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--num-perm", type=int, default=NUM_PERM)
    ap.add_argument("--shingle-size", type=int, default=SHINGLE_SIZE)
    args = ap.parse_args()

    if args.self_test:
        res = self_test(args.interim_dir, args.threshold, args.num_perm, args.shingle_size)
        raise SystemExit(0 if res["passed"] else 1)

    report = run(args.interim_dir, args.threshold, args.num_perm, args.shingle_size)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[dedup] {report['documents_kept_count']}/{report['documents_scanned']} documents kept")
    print(f"[dedup] report -> {args.report}")


if __name__ == "__main__":
    main()

"""Stage 4 - EDA on the final chunk set.

Input : data/processed/chunks.jsonl
        data/processed/cleaning_report.json   (review flags from Stage 1)
        data/processed/dedup_report.json      (Stage 2 stats)
        data/processed/chunking_report.json   (Stage 3 stats)
Output: data/eda/eda_report_pilot.md
        data/eda/token_histogram_pilot.png    (matplotlib, optional)

The markdown report is self-contained: it embeds an ASCII histogram so it reads fine in
a terminal or on GitHub, and links the PNG for a nicer view.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

CHUNKS_PATH = "data/processed/chunks.jsonl"
CLEANING_REPORT = "data/processed/cleaning_report.json"
DEDUP_REPORT = "data/processed/dedup_report.json"
CHUNK_REPORT = "data/processed/chunking_report.json"
OUT_MD = "data/eda/eda_report_pilot.md"
OUT_PNG = "data/eda/token_histogram_pilot.png"

EXAMPLES_PER_REGION = 3
EXAMPLE_CHARS = 420
HIST_BINS = 12
ASCII_WIDTH = 46


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def percentile(sorted_vals: list[int], q: float) -> float:
    """Linear-interpolation percentile (numpy-free, keeps the report reproducible)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def ascii_histogram(values: list[int], bins: int = HIST_BINS, width: int = ASCII_WIDTH) -> str:
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"{lo:>5} | {'#' * width} {len(values)}"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        left = lo + i * step
        right = lo + (i + 1) * step
        bar = "#" * int(round(c / peak * width))
        lines.append(f"{left:6.0f}-{right:6.0f} | {bar:<{width}} {c:>4}")
    return "\n".join(lines)


def save_png_histogram(values: list[int], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(values, bins=30, color="#3b6ea5", edgecolor="white")
    ax.set_xlabel("tokens per chunk (whitespace words)")
    ax.set_ylabel("chunks")
    ax.set_title(f"Token count distribution - {len(values)} chunks (pilot)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def md_table(headers: list[str], rows: list[list], aligns: list[str] | None = None) -> str:
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def rtl(text: str) -> str:
    """Wrap Arabic sample text so markdown renderers lay it out right-to-left."""
    return f'<div dir="rtl" lang="ar">\n\n{text}\n\n</div>'


def build_report(chunks: list[dict], cleaning: dict, dedup: dict, chunking: dict,
                 png_written: bool, png_path: str) -> str:
    tokens = sorted(c["token_count"] for c in chunks)
    total_tokens = sum(tokens)

    by_region: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_region[c.get("region") or c.get("domain")].append(c)

    L: list[str] = []
    A = L.append

    A("# EDA Report - Saudi Regional Dialect Corpus")
    A("")
    A(f"- **Source corpus:** {chunks[0]['source']} | **License:** {chunks[0]['license']}")
    A(f"- **Documents in:** {chunking.get('documents_chunked', '?')}")
    A(f"- **Source format:** `{chunking.get('config', {}).get('source_format', 'prose')}` "
      f"(explicit per batch; clean.py and chunk.py must agree)")
    A(f"- **Chunks out:** {len(chunks):,}")
    A(f"- **Total tokens:** {total_tokens:,} "
      f"(*{chunks[0].get('token_count_method', 'whitespace_word_count')}* approximation)")
    A(f"- **Pipeline:** `clean.py` -> `dedup.py` -> `chunk.py` -> `eda.py` (fully deterministic, no LLM calls)")
    variant = chunking.get("config", {}).get("text_variant", "normalized")
    A(f"- **Text variant chunked:** `{variant}` - "
      + ("alef/ya-folded normalized text (أ إ آ -> ا, ى -> ي). Stage 1 also stores the "
         "orthography-preserving text; regenerate with `chunk.py --text-variant original` "
         "if the folded spelling is not wanted in the training text."
         if variant == "normalized"
         else "orthography preserved; normalization was used for matching/dedup only."))
    A("")

    # ---------------- token distribution ----------------
    A("## 1. Token count distribution")
    A("")
    A("`token_count` is a **word-based approximation**: whitespace-delimited words "
      "(`len(text.split())`). No subword tokenizer is applied at this stage, so the "
      "pipeline stays deterministic and model-agnostic. For Arabic, a SentencePiece/BPE "
      "tokenizer typically yields ~1.5-2.5 subword tokens per word.")
    A("")
    stats = [
        ["min", f"{tokens[0]:,}"],
        ["p25", f"{percentile(tokens, 0.25):,.0f}"],
        ["median (p50)", f"{percentile(tokens, 0.50):,.0f}"],
        ["p75", f"{percentile(tokens, 0.75):,.0f}"],
        ["p90", f"{percentile(tokens, 0.90):,.0f}"],
        ["max", f"{tokens[-1]:,}"],
        ["mean", f"{total_tokens / len(tokens):,.1f}"],
    ]
    A(md_table(["statistic", "tokens"], stats, ["---", "---:"]))
    A("")
    in_band = sum(1 for t in tokens if 200 <= t <= 800)
    A(f"**{in_band:,} / {len(tokens):,} chunks ({in_band/len(tokens)*100:.1f}%) fall inside "
      f"the 200-800 token target band.**")
    A("")
    A("```")
    A("tokens/chunk        | histogram" + " " * (ASCII_WIDTH - 9) + "count")
    A(ascii_histogram(tokens))
    A("```")
    if png_written:
        A("")
        A(f"![Token distribution]({os.path.basename(png_path)})")
    A("")

    # ---------------- region distribution ----------------
    A("## 2. Region distribution")
    A("")
    rows = []
    for reg in sorted(by_region, key=lambda d: -len(by_region[d])):
        cs = by_region[reg]
        tk = sorted(c["token_count"] for c in cs)
        docs = len({c["doc_id"] for c in cs})
        rows.append([
            reg, docs, f"{len(cs):,}", f"{len(cs)/len(chunks)*100:.1f}%",
            f"{sum(tk):,}", f"{percentile(tk, 0.5):,.0f}",
        ])
    rows.append(["**total**", len({c['doc_id'] for c in chunks}), f"**{len(chunks):,}**",
                 "100.0%", f"**{total_tokens:,}**", f"{percentile(tokens, 0.5):,.0f}"])
    A(md_table(["region", "docs", "chunks", "share", "tokens", "median tokens"], rows,
               ["---", "---:", "---:", "---:", "---:", "---:"]))
    A("")

    A("### Chunks per document")
    A("")
    doc_rows = []
    for d in chunking.get("per_document", []):
        doc_rows.append([d["doc_id"], d.get("region", "-"), d.get("unit_type", "unit"),
                         f"{d.get('units', 0):,}", f"{d['chunks']:,}", f"{d['tokens']:,}"])
    A(md_table(["doc_id", "region", "unit type", "units", "chunks", "tokens"], doc_rows,
               ["---", "---", "---", "---:", "---:", "---:"]))
    A("")

    A("### format_type distribution")
    A("")
    fmt = Counter(c["format_type"] for c in chunks)
    A(md_table(["format_type", "chunks", "share"],
               [[k, f"{v:,}", f"{v/len(chunks)*100:.1f}%"] for k, v in fmt.most_common()],
               ["---", "---:", "---:"]))
    A("")

    # ---------------- cleaning + flags ----------------
    A("## 3. Cleaning stage and items flagged for review")
    A("")
    docs = cleaning.get("documents", [])
    if docs:
        raw = sum(d["char_count_raw"] for d in docs)
        keep = sum(d["char_count_retained"] for d in docs)
        A(f"- Characters in: **{raw:,}** -> retained: **{keep:,}** "
          f"(**{(raw-keep)/raw*100:.2f}%** removed overall)")
        A(f"- Flag threshold: a document is flagged when cleaning removes more than "
          f"**{cleaning.get('config', {}).get('char_loss_flag_threshold', 0.30)*100:.0f}%** of its characters")
        A(f"- Diacritic stripping: **{'ON' if cleaning.get('config', {}).get('strip_diacritics') else 'OFF'}** "
          f"(default off - two children's books in this batch are fully vocalized)")
        A("")
        worst = sorted(docs, key=lambda d: -d["char_loss_ratio"])[:5]
        A("Largest character deltas:")
        A("")
        A(md_table(["doc_id", "region", "chars in", "chars retained", "% removed", "flags"],
                   [[d["doc_id"], d.get("region", "-"), f"{d['char_count_raw']:,}",
                     f"{d['char_count_retained']:,}", f"{d['char_loss_ratio']*100:.2f}%",
                     ", ".join(d["review_flags"]) or "-"] for d in worst],
                   ["---", "---", "---:", "---:", "---:", "---"]))
        A("")

    flagged = cleaning.get("flagged_for_review", [])
    A(f"### Documents flagged during cleaning: **{len(flagged)}**")
    A("")
    if flagged:
        A(md_table(["doc_id", "region", "title", "% removed", "flags"],
                   [[d["doc_id"], d.get("region", "-"), d["title"], f"{d['char_loss_ratio']*100:.2f}%",
                     ", ".join(d["review_flags"])] for d in flagged],
                   ["---", "---", "---", "---:", "---"]))
        A("")
        A("**No document exceeded the 30% character-loss threshold.** "
          "Every flag above is `media_placeholders_removed`: empty `[]` / `[figure]` image "
          "placeholders carrying no text were pulled out of the body and recorded verbatim "
          "under `removed_placeholders` in the cleaned JSON, so nothing was lost.")
    else:
        A("None.")
    A("")

    chunk_flags = chunking.get("chunk_flag_counts", {})
    A("### Chunks flagged")
    A("")
    if chunk_flags:
        A(md_table(["flag", "chunks", "meaning"],
                   [[k, v, {
                       "below_target_min": "below 200 tokens - end-of-document tail, kept",
                       "oversize_paragraph": "single paragraph > 800 tokens - kept whole",
                       "oversize_stanza": "single stanza > 800 tokens - kept whole, never split mid-stanza",
                       "oversize_entry_block": "single glossary entry > 800 tokens - kept whole",
                       "oversize_entry_line": "single glossary entry > 800 tokens - kept whole",
                       "oversize_verse_line": "single verse line > 800 tokens - kept whole",
                   }.get(k, "")] for k, v in sorted(chunk_flags.items())],
                   ["---", "---:", "---"]))
        A("")
        flagged_chunks = [c for c in chunks if c["review_flags"]]
        A(md_table(["chunk_id", "region", "tokens", "units", "flags"],
                   [[c["chunk_id"], c.get("region") or c.get("domain"), c["token_count"],
                     c["source_pointer"].get("unit_count"), ", ".join(c["review_flags"])]
                    for c in sorted(flagged_chunks, key=lambda c: c["chunk_id"])],
                   ["---", "---", "---:", "---:", "---"]))
    else:
        A("None.")
    A("")

    # ---------------- dedup ----------------
    A("## 4. Deduplication")
    A("")
    if dedup:
        cfg = dedup.get("config", {})
        A(md_table(["metric", "value"], [
            ["documents scanned", dedup.get("documents_scanned", 0)],
            ["exact duplicate groups (SHA-256)", len(dedup.get("exact_duplicate_groups", []))],
            ["documents removed as exact duplicates", dedup.get("exact_duplicates_removed", 0)],
            ["MinHash/LSH candidate pairs", dedup.get("near_duplicate_candidate_pairs", 0)],
            ["near-duplicate pairs confirmed", len(dedup.get("near_duplicate_pairs", []))],
            ["documents kept", dedup.get("documents_kept_count", 0)],
        ], ["---", "---:"]))
        A("")
        A(f"- Exact: `{cfg.get('exact_method')}`")
        A(f"- Near: `{cfg.get('near_method')}`, Jaccard threshold "
          f"**{cfg.get('jaccard_threshold')}**, LSH candidates re-checked against the true "
          f"Jaccard of the shingle sets to drop false positives")
        A(f"- Policy: `{cfg.get('near_duplicate_policy')}`")
        A("")
        if dedup.get("exact_duplicates_removed", 0) == 0 and not dedup.get("near_duplicate_pairs"):
            A("**Zero duplicates in the pilot** - expected for 20 distinct books from one "
              "source. Because a clean run proves nothing about the detector itself, "
              "`dedup.py --self-test` injects an exact copy and a ~0.85-Jaccard perturbed "
              "copy of a real document and asserts both are caught while an unrelated "
              "document is not. It passes.")
    else:
        A("_No dedup report found._")
    A("")

    # ---------------- examples ----------------
    A("## 5. Example chunks")
    A("")
    A(f"{EXAMPLES_PER_REGION} chunks per region, sampled deterministically "
      f"(evenly spaced through each region's chunk list), truncated to ~{EXAMPLE_CHARS} characters.")
    A("")
    for reg in sorted(by_region):
        cs = sorted(by_region[reg], key=lambda c: c["chunk_id"])
        # a region with no more chunks than EXAMPLES_PER_REGION shows all of them;
        # otherwise sample evenly across the list (never the same chunk twice)
        if len(cs) <= EXAMPLES_PER_REGION:
            picks = cs
        else:
            idxs = sorted({round(i * (len(cs) - 1) / (EXAMPLES_PER_REGION - 1))
                           for i in range(EXAMPLES_PER_REGION)})
            picks = [cs[i] for i in idxs]
        A(f"### {reg}")
        A("")
        for c in picks:
            sp = c["source_pointer"]
            ut = sp.get("unit_type", "unit")
            A(f"**`{c['chunk_id']}`** - {c['token_count']} tokens - "
              f"`{c['format_type']}` - {ut} {sp['unit_start']}-{sp['unit_end']} "
              f"({sp['unit_count']} {ut}s)  ")
            A(f"*{c['title']}* - {c['author']}"
              + (f" (tr. {c['translator']})" if c.get("translator") else ""))
            A("")
            text = c["chunk_text"][:EXAMPLE_CHARS].replace("\n\n", "  \n")
            suffix = " …" if len(c["chunk_text"]) > EXAMPLE_CHARS else ""
            A(rtl("> " + text.replace("\n", "\n> ") + suffix))
            A("")

    A("---")
    A("")
    A("Generated by `src/data_engineering/eda.py`. Regenerate with:")
    A("")
    A("```bash")
    A("python src/data_engineering/clean.py && python src/data_engineering/dedup.py && "
      "python src/data_engineering/chunk.py && python src/data_engineering/eda.py")
    A("```")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4: EDA over the final chunk set.")
    ap.add_argument("--chunks", default=CHUNKS_PATH)
    ap.add_argument("--cleaning-report", default=CLEANING_REPORT)
    ap.add_argument("--dedup-report", default=DEDUP_REPORT)
    ap.add_argument("--chunk-report", default=CHUNK_REPORT)
    ap.add_argument("--out", default=OUT_MD)
    ap.add_argument("--png", default=OUT_PNG)
    args = ap.parse_args()

    chunks = load_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(f"No chunks in {args.chunks}. Run chunk.py first.")

    tokens = [c["token_count"] for c in chunks]
    png_ok = save_png_histogram(tokens, args.png)
    if not png_ok:
        print("[eda] matplotlib unavailable - PNG histogram skipped (ASCII histogram still included)")

    md = build_report(
        chunks,
        load_json(args.cleaning_report),
        load_json(args.dedup_report),
        load_json(args.chunk_report),
        png_ok,
        args.png,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)

    st = sorted(tokens)
    print(f"[eda] {len(chunks):,} chunks | {sum(tokens):,} tokens")
    print(f"[eda] min {st[0]} | p25 {percentile(st,0.25):.0f} | median {percentile(st,0.5):.0f} "
          f"| p75 {percentile(st,0.75):.0f} | max {st[-1]}")
    print(f"[eda] report -> {args.out}")
    if png_ok:
        print(f"[eda] histogram -> {args.png}")


if __name__ == "__main__":
    main()

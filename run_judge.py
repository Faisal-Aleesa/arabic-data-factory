# -*- coding: utf-8 -*-
import json
import sys
from arabic_dpo_judge import judge_pair

IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "dpo_pairs_classical_lexicon.jsonl"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "judge_results.jsonl"

rows = []
with open(IN_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

results = []
correct, total_scored = 0, 0

for i, row in enumerate(rows):
    r = judge_pair(row["prompt"], row["chosen"], row["rejected"], use_model_layer=True)
    stated_label = row.get("rejection_type")
    stated_meta_label = row.get("label")  # قد يحوي "BROKEN: ..." كتعليق بشري على العينة

    out = {
        "id": i,
        "source_chunk_id": row.get("source_chunk_id"),
        "prompt": row["prompt"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
        "dataset_stated_rejection_type": stated_label,
        "dataset_note": stated_meta_label,
        "judge_verdict": r["verdict"],
        "judge_rejection_type": r["rejection_type"],
        "judge_confidence": r["confidence"],
        "chosen_preferred_over_rejected": r["chosen_preferred_over_rejected"],
        "judge_layer_used": r["layer_used"],
        "judge_reasons": r["reasons"],
        "judge_evidence": r["evidence"],
    }
    results.append(out)

    # تقرير دقة ذاتي (لا يؤثر على قرار الجَحكم، فقط للعرض التحليلي)
    if r["rejection_type"] in {
        "poor_instruction_following", "wrong_formatting", "missing_information",
        "unsupported_additions", "wrong_register", "verbosity", "weak_organization",
        "partial_factual_errors", "less_faithful_reconstruction",
    }:
        total_scored += 1
        if r["rejection_type"] == stated_label:
            correct += 1

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"عدد الحالات: {len(results)}")
print(f"حالات صُنّفت ضمن التصنيفات القياسية: {total_scored}")
if total_scored:
    print(f"دقة تطابق مع التصنيف المعلن في البيانات: {correct}/{total_scored} = {correct/total_scored:.0%}")

verdict_counts = {}
for r in results:
    verdict_counts[r["judge_verdict"]] = verdict_counts.get(r["judge_verdict"], 0) + 1
print("توزيع الأحكام:", verdict_counts)

flagged = [r for r in results if r["judge_rejection_type"] in ("INVALID_PAIR_IDENTICAL", "SUSPECT_QUALITY_REVERSAL")]
print(f"\nحالات مُعطوبة اكتُشفت تلقائيا: {len(flagged)}")
for r in flagged:
    print(f"  - id={r['id']} dataset_note={r['dataset_note']!r} -> judge={r['judge_rejection_type']}")

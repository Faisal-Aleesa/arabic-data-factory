# ARCHIVED (2026-08-27): Hindawi is a Modern Standard Arabic book corpus, so it no longer fits the Saudi regional dialect phase; kept for provenance and to regenerate the archived MSA pilot under archive/msa_pilot/. Writes the pre-pivot manifest schema (no corpus_phase/region columns).
import json
import csv
import os
import random
from datetime import date
from datasets import load_dataset

# --- Category -> domain mapping ---
CATEGORY_MAP = {
    "science": "science_tech", "technology": "science_tech",
    "environmental.sciences": "science_tech", "health": "science_tech",
    "economics": "science_tech", "business": "science_tech", "psychology": "science_tech",
    "history": "administrative", "politics": "administrative",
    "social.sciences": "administrative", "biographies": "administrative",
    "religions": "administrative", "geography": "administrative",
    "travel.literature": "administrative",
    "children.stories": "educational",
    "arts": "general_knowledge", "philosophy": "general_knowledge",
    "literature": "general_knowledge", "novels": "general_knowledge",
    "poetry": "general_knowledge", "plays": "general_knowledge",
    "literary.criticism": "general_knowledge", "linguistics": "general_knowledge",
    "detective.fiction": "general_knowledge", "science.fiction": "general_knowledge",
}

SAMPLE_PER_DOMAIN = 5  # ~20 books total across 4 domains
BASE_DIR = "data/interim"
MANIFEST_PATH = "docs/license_manifest.csv"

print("Loading dataset (this downloads it the first time)...")
ds = load_dataset("mohres/The_Arabic_E-Book_Corpus", split="train")
print(f"Loaded {len(ds)} books.")

# Group row indices by mapped domain
by_domain = {"science_tech": [], "administrative": [], "educational": [], "general_knowledge": []}
for i, row in enumerate(ds):
    domain = CATEGORY_MAP.get(row["category.main"])
    if domain:
        by_domain[domain].append(i)

random.seed(42)
manifest_rows = []

for domain, indices in by_domain.items():
    os.makedirs(f"{BASE_DIR}/{domain}", exist_ok=True)
    sample_size = min(SAMPLE_PER_DOMAIN, len(indices))
    chosen = random.sample(indices, sample_size)
    print(f"{domain}: sampling {sample_size} of {len(indices)} available")

    for idx in chosen:
        row = ds[idx]
        doc_id = str(row["booknr"])
        out_path = f"{BASE_DIR}/{domain}/{doc_id}.json"

        record = {
            "doc_id": doc_id,
            "title": row["title"],
            "author": row["author"],
            "domain": domain,
            "category_original": row["category"],
            "source": "Hindawi via Arabic E-Book Corpus (HF)",
            "license": "CC-BY-4.0",
            "word_count": row["wc"],
            "raw_text": row["text"],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        manifest_rows.append({
            "doc_id": doc_id,
            "title": row["title"],
            "source_site": "Hugging Face (mohres/The_Arabic_E-Book_Corpus)",
            "url": "https://huggingface.co/datasets/mohres/The_Arabic_E-Book_Corpus",
            "license": "CC-BY-4.0",
            "license_note": "Hallberg, A. (2024). The Arabic E-Book Corpus. University of Gothenburg.",
            "domain": domain,
            "format": "text (parquet)",
            "date_checked": str(date.today()),
        })

os.makedirs("docs", exist_ok=True)
write_header = not os.path.exists(MANIFEST_PATH)
with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
    if write_header:
        writer.writeheader()
    writer.writerows(manifest_rows)

print(f"\nDone. {len(manifest_rows)} books saved across {len(by_domain)} domains.")
print(f"Manifest updated at {MANIFEST_PATH}")
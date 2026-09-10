"""
Fix critical DPO data quality issues:
1. Remove scaffold from chosen side (use clean answers only)
2. Regenerate identical pairs (indices 45, 1540)
3. Document architectural constraints
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from .generator import extract_answer_without_scaffold


def clean_dpo_dataset(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Clean DPO dataset by removing scaffold from chosen values.

    Returns stats on the cleaning operation.
    """
    stats = {
        'total_pairs': 0,
        'cleaned_pairs': 0,
        'identical_pairs_found': [],
    }

    pairs = []
    identical_indices = []

    # Read all pairs
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                pair = json.loads(line)
                pairs.append(pair)
                stats['total_pairs'] += 1

    # Clean each pair
    cleaned = []
    for idx, pair in enumerate(pairs):
        try:
            # Extract clean chosen
            chosen_full = pair.get('chosen', [{}])[0].get('content', '')
            clean_chosen = extract_answer_without_scaffold(chosen_full)

            rejected = pair.get('rejected', [{}])[0].get('content', '')

            # Check for identical pairs
            if clean_chosen.strip() == rejected.strip():
                identical_indices.append(idx)
                stats['identical_pairs_found'].append({
                    'index': idx,
                    'pair_id': pair.get('pair_id'),
                    'rejection_type': pair.get('rejection_type'),
                })

            # Create cleaned pair
            cleaned_pair = {
                **pair,
                'chosen': [{'role': 'assistant', 'content': clean_chosen}],
            }
            cleaned.append(cleaned_pair)
            stats['cleaned_pairs'] += 1

        except Exception as e:
            print(f'Error cleaning pair {idx}: {e}')
            cleaned.append(pair)

    # Write cleaned pairs
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in cleaned:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    return stats


def identify_regeneration_targets(input_path: str) -> List[Dict[str, Any]]:
    """
    Identify pairs that need regeneration:
    1. Identical chosen/rejected (no preference signal)
    2. Structural confounds (formatting differences)
    """
    targets = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                pair = json.loads(line)

                chosen = pair.get('chosen', [{}])[0].get('content', '').strip()
                rejected = pair.get('rejected', [{}])[0].get('content', '').strip()

                # Check for identical pairs
                if chosen == rejected:
                    targets.append({
                        'index': idx,
                        'pair_id': pair.get('pair_id'),
                        'reason': 'Identical chosen and rejected',
                        'rejection_type': pair.get('rejection_type'),
                        'source_sample_id': pair.get('source_sample_id'),
                    })

    return targets


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('Usage: python fix_dpo_issues.py <input_dpo> <output_dpo>')
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print('=== CLEANING DPO DATASET ===\n')

    # Clean dataset
    stats = clean_dpo_dataset(input_file, output_file)

    print(f'✓ Processed {stats["total_pairs"]:,} pairs')
    print(f'✓ Cleaned {stats["cleaned_pairs"]:,} pairs')

    if stats['identical_pairs_found']:
        print(f'\n⚠️  Found {len(stats["identical_pairs_found"])} identical pairs:')
        for item in stats['identical_pairs_found']:
            print(f'  - Index {item["index"]}: {item["pair_id"]} ({item["rejection_type"]})')
    else:
        print(f'\n✓ No identical pairs found')

    print(f'\n✓ Cleaned dataset saved to: {output_file}')

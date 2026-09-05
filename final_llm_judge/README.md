# Arabic LLM Judge — Quality Evaluation Dashboard

`dashboard.html` — open directly in any browser, no server or build step needed.
Data is embedded inline from `dashboard_data.json`.

## What this dashboard is, precisely

It renders the 9 charter rejection types against one real fixture chunk
(`atc_c0001`, from `tests/fixtures/chunks_classical_lexicon.jsonl`), showing:

- the Unified Verdict per case (`AUTO_CONFIRM` / `NEEDS_JUDGE` / `FLAG_SUSPICIOUS`),
  copied from `src/dpo/test_llm_judge.py`'s `EXPECTED_VERDICT` and confirmed by
  `test_report.txt`'s layer_B PASS rows against the real `check_dpo.check_pair()`
- for every `NEEDS_JUDGE` case: the actual `judge assessment`, `confidence`,
  `evidence` (claim + literal quote + source), and `type_scores`, produced by
  **actually running** `judge_contract.py` / `judge_client.py` / `llm_judge.py`
  (see `src/dpo/generate_dashboard_data.py`) — not hand-typed JSON
- `dialect_markers_in_rejected` as a clearly labeled **supporting signal only**,
  computed by the new `src/dpo/dialect_markers.py` module, on the real rejected
  text of each case — never used by the Judge or `check_dpo` to decide a verdict

## Two honest, stated limitations

1. **No live LLM in this sandbox.** No network access, no `JUDGE_API_KEY`. The six
   `NEEDS_JUDGE` rows use `judge_client.RecordedJudgeClient` (offline replay) — the
   same no-network path the project's own test suite uses. `judge_model_version` is
   left as `recorded/offline`, never disguised as a live call. Wiring
   `AnthropicJudgeClient` (see `README.md`'s "Wiring a real provider") and re-running
   `generate_dashboard_data.py` turns every row into a live Judge call with no
   structural change.

2. **`check_dpo.py` / `verify_sft.py` (Phase 1) are not part of this delivery
   bundle** (per the top-level `README.md`: "imported, never copied"), so
   deterministic/similarity `chosen_pct` / `rejected_pct` / `distinctness` /
   `format_check` numbers could not be computed here. The dashboard marks those
   fields `available: false` with a stated reason instead of showing invented
   numbers.

## Regenerating the data

```bash
cd src/dpo
python3 generate_dashboard_data.py   # writes ../../final_llm_judge/dashboard_data.json
```

Then re-embed it into `dashboard.html` (replace the JSON inside the
`<script id="judge-data" type="application/json">` tag) or open
`dashboard.html` next to a fresh `dashboard_data.json` and change the script tag
to `fetch('dashboard_data.json')` if you prefer it un-embedded for a server context.

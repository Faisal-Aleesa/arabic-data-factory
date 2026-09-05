# Judge endpoint setup

**إعداد نقطة النهاية للحَكَم: أربعة متغيرات بيئة، ولا شيء غيرها.**

The LLM judge stack (`src/verification/judge_*.py`) is complete and self-tested. It has
never made a real API call, because no endpoint or credential exists yet. This file is
what someone with those details needs, and nothing here should be filled in by anyone who
does not own the endpoint.

**Nothing in this repo contains a base URL, a model id, or a key.** That is deliberate.
A `--self-test` on every module passes without any of them.

---

## The four variables

| variable | meaning | example shape |
|---|---|---|
| `JUDGE_API_BASE_URL` | endpoint base, no trailing slash | `https://<host>/compatible-mode/v1` |
| `JUDGE_API_MODEL` | model id the endpoint expects | `qwen-<variant>` |
| `JUDGE_API_KEY_VAR` | **the NAME of the variable holding the key**, not the key | `DASHSCOPE_API_KEY` |
| `JUDGE_API_DIALECT` | optional; `openai_chat` (default), `anthropic_messages`, `custom` | `openai_chat` |

`JUDGE_API_KEY_VAR` is a level of indirection on purpose. The key itself is read from the
variable it names, at call time, and is never stored on any object, never written to a
transcript, and stripped from every error message by `_redact()`. A self-test asserts the
key does not appear in an exception even when the endpoint echoes it back in an error
body — which some gateways do.

```bash
export JUDGE_API_BASE_URL="https://<host>/compatible-mode/v1"
export JUDGE_API_MODEL="<model-id>"
export JUDGE_API_KEY_VAR="MY_PROVIDER_KEY"
export MY_PROVIDER_KEY="<the actual key>"      # set by whoever owns the credential
```

Check it resolved, without printing the key:

```bash
python src/verification/judge_backend_http.py --describe
```

That prints the base URL, model, dialect, and `api_key_present: true|false`. It never
prints the key.

## Which dialect

`openai_chat` is the default and covers most hosted Qwen deployments — DashScope's
OpenAI-compatible mode, vLLM, Ollama's OpenAI shim, Together, OpenRouter, LM Studio, and
most self-hosted gateways. It POSTs `{base}/chat/completions` with a `messages` array and
reads `choices[0].message.content`.

`anthropic_messages` exists only because that SDK happens to be installed in this
project's environment. The wire shape genuinely differs (system is a top-level field,
content is a block list), and confusing the two yields an empty string rather than an
error.

`custom` handles anything else without code changes — give `text_path` and optionally
`usage_in_path` / `usage_out_path` as key sequences into the response JSON:

```python
EndpointConfig(base_url=..., model=..., api_key_var=...,
               dialect='custom', path='/generate',
               text_path=['output', 'text'],
               usage_in_path=['meta', 'in'], usage_out_path=['meta', 'out'])
```

## First run, in order

```bash
# 1. structure only, no model, no cost
python src/verification/judge_suite.py --validate

# 2. what a full run will cost, using the provider's real per-MTok rates
python src/verification/judge_suite.py --dry-run --price-in <IN> --price-out <OUT>

# 3. the real run, recording every answer so it can be replayed free
python src/verification/judge_suite.py --run --backend http \
    --transcript data/judge_transcript.jsonl \
    --price-in <IN> --price-out <OUT>

# 4. re-score offline from the transcript, no further spend
python src/verification/judge_suite.py --run --backend http \
    --transcript data/judge_transcript.jsonl --replay-only
```

Step 3 is the first time this project spends money. Measured on the current fixtures the
suite is **29 calls / ~68,500 input tokens / ~8,700 output tokens** — 24 probes plus 5
position-swap passes.

> **`data/judge_transcript.jsonl` must be gitignored before step 3.** It records prompts,
> and a prompt embeds the full source chunk. A transcript over dialect probes therefore
> inherits the dialect corpus's rights-pending status, exactly like
> `tests/fixtures/*saudi_dialect*`. Add the ignore rule in the same commit that creates
> the file, per the rule in the session handoff.

## Reading the result

The matrix scores each probe against the expectation recorded in the fixture. Two numbers
matter more than the headline accuracy:

- **SFT-06 (gameability)** — a polished, confident response with a formulaic
  source-citing clause over invented content. The same shape moved `check_similarity`
  from pct 31.9 to 90.8 and produced a false PASS. If the judge passes SFT-06, it shares
  that blind spot and must never be a sole gate.
- **SFT-07 (discrimination)** — SFT-01's clean response pointed at the wrong chunk. A
  judge that passes both is not reading the SOURCE, and every other PASS it gives is
  worthless.

An all-abstain run prints `NO PROBE WAS SCORED` and no accuracy figure. That is a backend
or prompt failure, not a judge result — do not read it as 0%.

## What a green run still does not establish

`wrong_register` and `weak_organization` have **zero** automated coverage anywhere in the
project (measured, `check_dpo.DETECTABILITY`). For those two types the judge is the only
verification that exists, so there is nothing to cross-validate it against even with a
real model. The position-swap check catches a judge tracking slot order, but it cannot
tell you the judge understands Arabic register. **Those two types need human spot-checks
before the judge is trusted at scale on them.**

Fixtures are classical-only, so they are committable under CC BY-SA. Dialect register is
a different axis (dialectal vs MSA, not classical vs casual) and needs its own probes,
built locally and gitignored.

## After the first real run

The combination functions `judge_sft.combine_with_rules()` and
`judge_dpo.combine_with_dpo_rules()` implement the disagreement table in
`src/deployment/ACCEPTANCE_CONTRACT.md` and are self-tested on every cell — but they are
**not wired** into `verify_sft.acceptance_decision()`, which still knows nothing about a
judge. Wire them only after real judge output has been scored here. Every threshold in
this project that was set before meeting real data had to be corrected afterwards.

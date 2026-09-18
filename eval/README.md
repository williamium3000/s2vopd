# S2VOPD Evaluation Suite

The evaluation harness used for all S2VOPD results. It talks to any **OpenAI-compatible
endpoint**, so the policy can be served by vLLM, SGLang, or a hosted API — the harness itself
never loads model weights.

Pipeline per benchmark: `prepare_data.py` → `infer.py` → `judge_qwenlm.py` → `cal_acc.py`.
`run_eval.sh` chains all four.

## Install

```bash
pip install -r eval/requirements.txt
```

## Quick start

Serve the policy (any OpenAI-compatible server), then:

```bash
API_BASE="http://localhost:8000/v1/" \
OPENAI_MODEL_ID="S2VOPD-Qwen3.5-4B" \
BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
bash eval/run_eval.sh
```

Benchmark data is downloaded from the Hugging Face Hub on first use and cached next to the
scripts. Results land in `model_answer/`, judge outputs in `judge/`, and accuracy is printed
per benchmark.

## Protocols

Two protocols were used for the published numbers. **Do not mix them** — the same checkpoint
scores differently under each.

**Fine-grained perception** — greedy decoding, no sampling:

```bash
API_BASE=... OPENAI_MODEL_ID=... SEED=42 MAX_TOKENS=4096 \
BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
bash eval/run_eval.sh
```

**Mathematical reasoning** — long budget with sampling:

```bash
API_BASE=... OPENAI_MODEL_ID=... SEED=42 \
MAX_TOKENS=24576 TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 PRESENCE_PENALTY=1.5 \
BENCHMARK="mathvista,mathverse,mathvision,wemath" \
JUDGE_API_BASE="http://localhost:8001/v1/" JUDGE_MODEL="Qwen2.5-72B-Instruct" \
bash eval/run_eval.sh
```

The long budget matters: at 16k tokens, 8–11% of generations still hit the cap on the math
suite, which depresses scores. Math numbers produced under a 4096-token greedy protocol are
not comparable to the ones reported for S2VOPD.

## Judging

Multiple-choice and short-answer benchmarks are scored by an LLM judge
(**Qwen2.5-72B-Instruct** for all published numbers). Point the judge at a served endpoint:

```bash
JUDGE_API_BASE="http://localhost:8001/v1/" JUDGE_MODEL="Qwen2.5-72B-Instruct"
```

or run it locally through vLLM with `JUDGE_MODEL_PATH=/path/to/Qwen2.5-72B-Instruct`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `API_BASE` | *(required)* | OpenAI-compatible endpoint serving the policy |
| `OPENAI_MODEL_ID` | *(required)* | `served-model-name` of the policy |
| `BENCHMARK` | `vstar` | Comma-separated benchmark list |
| `SEED` | `42` | Sampling seed |
| `MAX_TOKENS` | `32768` | Generation budget |
| `TEMPERATURE` / `TOP_P` / `TOP_K` / `PRESENCE_PENALTY` | unset (greedy) | Sampling controls |
| `PARALLEL_WORKERS` | `256` | Concurrent requests |
| `MAX_RETRIES` | `3` | Retries per sample |
| `OUT_DIR` | `model_answer` | Where raw generations go |
| `JUDGE_API_BASE` / `JUDGE_MODEL` | unset | Judge served over an API |
| `JUDGE_MODEL_PATH` | unset | Judge loaded locally via vLLM |
| `JUDGE_MAX_TOKENS` | `2048` | Judge generation budget |

## Supported benchmarks

**Fine-grained perception** — `vstar`, `zoombench`, `hrbench-4k`, `hrbench-8k`,
`mme-realworld`, `mme-realworld-cn`, `mme-realworld-lite`, `visualprobe`, `cv-bench`, `mmvp`

**Mathematical reasoning** — `mathvista`, `mathverse`, `mathvision`, `mathvision-op`, `wemath`

**General multimodal** — `mmstar`, `realworldqa`, `logicvista`, `mmbench`, `mmbench-v11`,
`scienceqa-img`, `blink`, `hallusionbench`

**Hallucination** — `pope`, `pope_adv`, `pope_pop`, `pope_random`

`orchestrate_mmeval.sh` is a convenience wrapper that runs the seven general multimodal
benchmarks in sequence and writes a summary file.

## Files

| File | Role |
|---|---|
| `run_eval.sh` | Entry point; chains prepare → infer → judge → accuracy |
| `prepare_data.py` | Downloads each benchmark from the Hub and converts it to the common JSON schema |
| `infer.py` | Queries the served policy; handles retries, parallelism, multi-image inputs |
| `judge_qwenlm.py` | LLM judge, over an API or a local vLLM instance |
| `cal_acc.py` | Per-benchmark accuracy (each suite has its own scoring rule) |
| `mmeval_bench.py` | Builds the general multimodal benchmarks with VLMEvalKit-aligned prompts |
| `orchestrate_mmeval.sh` | Batch wrapper for the seven general multimodal benchmarks |

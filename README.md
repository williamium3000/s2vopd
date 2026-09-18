# S2VOPD

**Self-Supervised Visual On-Policy Distillation** — teaching multimodal LLMs to see fine detail
without rewards, human annotation, or a larger teacher.

A student model rolls out on a *degraded* view of an image while an EMA teacher scores the same
prefix on the *clean* original. The gap between the two is the entire training signal: it is
large exactly on the tokens that depend on visual detail, and near zero everywhere else.

## Models

| Model | Perception (6) | Math (4) |
|---|---|---|
| [S2VOPD-Qwen3.5-4B](https://huggingface.co/S2Visual-OPD/S2VOPD-Qwen3.5-4B) | 76.94 | 76.32 |
| [S2VOPD-Qwen3.5-9B](https://huggingface.co/S2Visual-OPD/S2VOPD-Qwen3.5-9B) | 77.81 | 78.65 |

Perception = V\* Bench, ZoomBench, HR-Bench 4K/8K, MME-RealWorld EN/CN (greedy, 4096 tokens).
Math = MathVista, MathVerse, MathVision, WeMath (24576 tokens, T=0.3, judged by Qwen2.5-72B-Instruct).
Per-benchmark breakdowns are on the model cards.

## What is released

- **[`eval/`](eval/)** — the full evaluation suite used for every number above. It drives any
  OpenAI-compatible endpoint, downloads each benchmark from the Hub, runs the LLM judge, and
  reports per-benchmark accuracy. See [`eval/README.md`](eval/README.md).

Training code and data release to follow.

## Quick start

```bash
pip install -r eval/requirements.txt

# serve the policy with any OpenAI-compatible server, then:
API_BASE="http://localhost:8000/v1/" \
OPENAI_MODEL_ID="S2VOPD-Qwen3.5-4B" \
SEED=42 MAX_TOKENS=4096 \
BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
bash eval/run_eval.sh
```

## Training data

[Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K) — 6,241 examples;
models above are trained for one epoch (65 steps).

## License

Apache-2.0.

<div align="center">

# S²VOPD

### Official implementation of [*Self-Supervised Visual On-Policy Distillation*](https://arxiv.org/abs/2608.14144)

[![arXiv](https://img.shields.io/badge/arXiv-2608.14144-b31b1b.svg)](https://arxiv.org/abs/2608.14144)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://williamium3000.github.io/s2vopd/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-S2Visual--OPD-yellow.svg)](https://huggingface.co/S2Visual-OPD)

Yijiang Li · Yijun Liang · Yunjie Tian · Bingyang Wang · Ke Zhang · Zhenfei Yin · Di Fu · Philip Torr · Nuno Vasconcelos

</div>

---

Teaching multimodal LLMs to see fine detail **without rewards, human annotation, or a larger teacher**.

A student model generates on a *degraded* view of an image while an EMA teacher scores the same
prefix on the *clean* original. The disagreement between the two is the entire training signal: it is
large exactly on the tokens that depend on visual detail, and near zero everywhere else.

## Links

| | |
|---|---|
| Paper | https://arxiv.org/abs/2608.14144 |
| Project page | https://williamium3000.github.io/s2vopd/ |
| Code | https://github.com/williamium3000/s2vopd |
| Models | https://huggingface.co/S2Visual-OPD |
| Training data | [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K) |

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

## Citation

```bibtex
@misc{li2026selfsupervisedvisualonpolicydistillation,
  title={Self-Supervised Visual On-Policy Distillation},
  author={Yijiang Li and Yijun Liang and Yunjie Tian and Bingyang Wang and Ke Zhang and Zhenfei Yin and Di Fu and Philip Torr and Nuno Vasconcelos},
  year={2026},
  eprint={2608.14144},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.14144},
}
```

## License

Apache-2.0.

<div align="center">

# S²VOPD

### Official implementation of [*Self-Supervised Visual On-Policy Distillation*](https://arxiv.org/abs/2608.14144)

[![arXiv](https://img.shields.io/badge/arXiv-2608.14144-b31b1b.svg)](https://arxiv.org/abs/2608.14144)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://williamium3000.github.io/s2vopd/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-s2vopd-yellow.svg)](https://huggingface.co/s2vopd)

Yijiang Li, Yijun Liang, Yunjie Tian, Bingyang Wang, Ke Zhang, Zhenfei Yin, Di Fu, Philip Torr, Nuno Vasconcelos

</div>

---

## Abstract

Visual on-policy distillation relies heavily on an informative teacher-student asymmetry, through
either a larger, stronger teacher or privileged supervision, such as reference answers or
ground-truth regions of interest. This raises a fundamental question: where can informative
asymmetry come from when nothing privileged is available? We answer this by inverting where the
asymmetry comes from. Rather than adding privileged information to the teacher, we subtract
information from the student. This asymmetry creates the same effective learning signal for free as
a teacher with access to information unavailable to the student, without ground-truth annotations,
rewards, or a separate stronger teacher model. Building on this principle, we introduce
Self-Supervised Visual On-Policy Distillation (S²VOPD), a simple yet effective method that
constructs on-policy learning signals from asymmetric augmented views. S²VOPD distills the teacher's
distribution conditioned on the original image on-policy into the student distribution conditioned
on a strongly augmented view of the same image. We systematically explore a broad design space of
visual augmentations and uncover that (1) **asymmetry matters**: all four augmentation families
improve performance, while symmetric self-distillation degrades it; (2) **strength matters**:
performance peaks at a moderate strength; and (3) **the gap must remain task-consistent**:
augmentations that completely remove the question-relevant evidence can induce large but
uninformative discrepancies.

## Models

| Model | Perception (6) | Math (4) |
|---|---|---|
| [S2VOPD-Qwen3.5-4B](https://huggingface.co/s2vopd/S2VOPD-Qwen3.5-4B) | 76.94 | 76.32 |
| [S2VOPD-Qwen3.5-9B](https://huggingface.co/s2vopd/S2VOPD-Qwen3.5-9B) | 77.81 | 78.65 |

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

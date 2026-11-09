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
| [S2VOPD-Qwen3.5-4B](https://huggingface.co/s2vopd/S2VOPD-Qwen3.5-4B) | 77.76 | 76.50 |
| [S2VOPD-Qwen3.5-4B-Contrastive](https://huggingface.co/s2vopd/S2VOPD-Qwen3.5-4B-Contrastive) | **78.75** | **78.88** |
| [S2VOPD-Qwen3.5-9B](https://huggingface.co/s2vopd/S2VOPD-Qwen3.5-9B) | **78.26** | **78.84** |

Gains over the corresponding base model:

| | Perception (6) | Math (4) |
|---|---|---|
| Qwen3.5-4B base | 71.39 | 76.99 |
| + S²VOPD | 77.76 `+6.37` | 76.50 |
| + S²VOPD & contrast | 78.75 `+7.36` | 78.88 `+1.89` |
| Qwen3.5-9B base | 74.58 | 78.89 |
| + S²VOPD | 78.26 `+3.68` | 78.84 |

Perception = V\* Bench, ZoomBench, HR-Bench 4K/8K, MME-RealWorld EN/CN (greedy, 4096 tokens).
Math = MathVista, MathVerse, MathVision, WeMath (24576 tokens, T=0.3, top-p 0.95, judged by
Qwen2.5-72B-Instruct). All numbers are one training epoch (65 steps). Per-benchmark breakdowns are
on the model cards.

## Method

S²VOPD manufactures the teacher-student asymmetry on the **input** side instead of the model side:

1. **Student view** — the image is augmented (information is *removed*).
2. **Teacher view** — the same model, kept as an EMA shadow of the student, sees the **clean** image.
3. **Objective** — on-policy self-distillation (JSD, `alpha=0.5`) of the teacher's next-token
   distribution over the student's own rollouts. No external teacher, no labels, no reward verifier.

Optionally (`self_distillation.contrastive_target=True`) the target is **contrast-sharpened**: the teacher is additionally
scored on a black canvas and the distillation target becomes

```
softmax( anchor * log p_teacher(clean) + alpha * [ log p_teacher(clean) - log p_teacher(black) ] )
```

restricted to the teacher's plausible support `{w : p_teacher(w) >= beta * max p_teacher}`. This
isolates what the image contributes and tilts the target towards image-dependent tokens.

## What is released

- **[`verl/`](verl/)** — the training framework with the S²VOPD additions: the augmentation space,
  the dual-view data path (student augmented / teacher clean), the EMA self-teacher, and the
  optional contrast-sharpened target. Qwen3.5, Qwen3-VL and Qwen2.5-VL are supported.
- **[`aug_configs/default_aug.json`](aug_configs/default_aug.json)** — the default augmentation
  recipe used for every number above.
- **[`scripts/`](scripts/)** — data preparation, the four training entry points, and checkpoint merging.
- **[`eval/`](eval/)** — the full evaluation suite used for every number above. See
  [`eval/README.md`](eval/README.md).

## Getting started

The training stack and the evaluation suite are independent: training needs GPUs, vLLM and
FlashAttention, while the evaluation suite only talks to an OpenAI-compatible endpoint. Install
them in separate environments.

### 1. Training environment

```bash
python3 -m venv .venv                        # Python >= 3.10
.venv/bin/pip install -U pip wheel setuptools

# requirements.txt is a complete freeze of the environment the reported runs used, so it is
# restored with --no-deps. Letting pip re-resolve it fails: a few packages declare metadata
# that contradicts the pinned numpy.
.venv/bin/pip install --no-deps -r requirements.txt

# FlashAttention is NOT in requirements.txt: it has to be built against the torch that is
# already installed, so it cannot be resolved in the same pass. Install it afterwards.
# Building it (and vLLM's JIT kernels for Qwen3.5) needs a CUDA toolkit >= 12.8 on PATH.
MAX_JOBS=32 .venv/bin/pip install --no-build-isolation flash-attn
```

`pip check` will report a few version mismatches (for example vLLM asking for
`transformers<5`). They are expected: the Qwen3.5 runs need `transformers==5.5.0`, and the
pinned set is the one the reported numbers were produced with.

### 2. Training data

```bash
.venv/bin/python scripts/prepare_data.py --data-dir ./data
```

Downloads [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K) and writes
`data/train.parquet`. The box-free setting only reads the original images, so only those are
fetched (~6 GB of the 37 GB repository). The red-box hint is stripped from every question.

### 3. Train

```bash
bash scripts/train_4b_s2vopd.sh               # Qwen3.5-4B, S²VOPD
bash scripts/train_4b_s2vopd_contrastive.sh   # Qwen3.5-4B, S²VOPD + contrastive target
bash scripts/train_9b_s2vopd.sh               # Qwen3.5-9B, S²VOPD
bash scripts/train_9b_s2vopd_contrastive.sh   # Qwen3.5-9B, S²VOPD + contrastive target
```

Each script runs one epoch (65 steps) on 8 GPUs and saves every 13 steps. `MODEL_PATH` points at
a local checkpoint or a Hub id, and any Hydra override can be appended, for example
`bash scripts/train_4b_s2vopd.sh trainer.total_training_steps=3` for a short smoke run. Merge an
FSDP checkpoint into HuggingFace format with `bash scripts/merge_checkpoint.sh <checkpoint_dir>`.

The augmentation recipe is a JSON file: `AUG_CONFIG=/path/to/recipe.json` overrides the default
and `AUG_PROB` sets the per-sample probability (1.0 in all reported runs). With `AUG_PROB=1.0`
every sample goes through the recipe, but the default recipe's operator weights sum to 0.9, so
about 10% of samples are still left unaugmented by design. The operator space lives in
[`verl/utils/dataset/image_aug.py`](verl/utils/dataset/image_aug.py).

The reported hyperparameters (lr 2e-6, batch 96, rollout n=8, ...) are set by
`scripts/run_s2vopd.sh`; the defaults in `verl/trainer/config/vopd.yaml` alone do not reproduce
the runs.

### 4. Evaluation environment

```bash
python3 -m venv .venv-eval
.venv-eval/bin/pip install -r eval/requirements.txt
```

Serve the policy with any OpenAI-compatible server (vLLM, SGLang, ...), then run the two
protocols. They are not interchangeable — the same checkpoint scores differently under each.
Every benchmark needs a judge: answers are graded by rule first (option letter, symbolic
math) and fall back to an LLM judge, so serve the judge (we use Qwen2.5-72B-Instruct) on its own
endpoint before either command, e.g. `vllm serve Qwen/Qwen2.5-72B-Instruct --port 8001
--tensor-parallel-size 2`.

```bash
# Fine-grained perception: greedy, 4096 tokens
API_BASE="http://localhost:8000/v1/" OPENAI_MODEL_ID="S2VOPD-Qwen3.5-4B" \
SEED=42 MAX_TOKENS=4096 \
BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
JUDGE_API_BASE="http://localhost:8001/v1/" JUDGE_MODEL="Qwen2.5-72B-Instruct" \
bash eval/run_eval.sh

# Mathematical reasoning: long budget with sampling, LLM judge
API_BASE="http://localhost:8000/v1/" OPENAI_MODEL_ID="S2VOPD-Qwen3.5-4B" \
SEED=42 MAX_TOKENS=24576 TEMPERATURE=0.3 TOP_P=0.95 TOP_K=20 PRESENCE_PENALTY=1.5 \
BENCHMARK="mathvista,mathverse,mathvision,wemath" \
JUDGE_API_BASE="http://localhost:8001/v1/" JUDGE_MODEL="Qwen2.5-72B-Instruct" \
bash eval/run_eval.sh
```

WeMath is built from VLMEvalKit's `WeMath.tsv` (the exact prompts of the reported numbers); it
is not fetched automatically. Place it at `~/LMUData/WeMath.tsv` (or point `LMUData` at its
directory) and check `md5sum` = `b5e969a075f01290a542411fb7766388`; the file is listed in
VLMEvalKit's `image_mcq.py`.

Generations are cached per model name and benchmark, and a rerun resumes from the cached file
whatever the decoding settings; use a fresh `MODEL_NAME` or `OUT_DIR` when you change
`MAX_TOKENS` or the sampling parameters.

See [`eval/README.md`](eval/README.md) for the per-benchmark details and every environment
variable.

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

Apache-2.0. This repository builds on [verl](https://github.com/volcengine/verl); see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

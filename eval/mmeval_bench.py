"""Prepare mm-eval/* HuggingFace benchmarks into Vision-OPD eval JSON format.

Image source = simple-mmeval's `mm-eval/*` HF datasets (same images/questions the
simple-mmeval OPSD eval uses). Prompt is rendered with simple-mmeval's
`vlmevalkit_mcq.txt` template (VLMEvalKit-aligned) so numbers line up with the
official-style report. Output rows follow the existing Vision-OPD schema:
  {"images": [path,...], "query": str, "response": str, "question_id": ..., "category": str}

Single-image benchmarks (all media=1): mmstar, realworldqa, logicvista, mmbench,
mmbench-v11, scienceqa-img. BLINK has multi-image sub-tasks -> all media kept in
`images` (needs infer.py multi-image support).
"""
import json
import re
from pathlib import Path

from jinja2 import Template
from tqdm import tqdm

# Verbatim from simple-mmeval/scripts/templates/vlmevalkit_mcq.txt (the template the
# OPSD eval passes via --template). Renders: question + (Options: A. .. B. ..) + instruction.
# NOTE: this template does NOT render `hint` (matches simple-mmeval's OPSD run).
_VLMEVALKIT_MCQ_TEMPLATE = (
    "{{ question }}{% if options %}\n"
    "Options:\n"
    "{% for k, v in options.items() %}{{ k }}. {{ v }}\n"
    "{% endfor %}{% endif %}\n"
    "Please select the correct answer from the options above."
)
_TMPL = Template(_VLMEVALKIT_MCQ_TEMPLATE)

# benchmark key -> (hf_repo, config, split). config auto-resolves if absent.
MMEVAL_HF_SPECS = {
    "mmstar":        ("mm-eval/MMStar",        "default", "val"),
    "realworldqa":   ("mm-eval/RealWorldQA",   "default", "test"),
    "logicvista":    ("mm-eval/LogicVista",    "default", "test"),
    "mmbench":       ("mm-eval/MMBench",       "en",      "dev"),
    "mmbench-v11":   ("mm-eval/MMBench-V11",   "en",      "dev"),
    "scienceqa-img": ("mm-eval/ScienceQA-IMG", "default", "validation"),
}

# BLINK: 14 sub-task configs, split=val (metric = pooled/micro accuracy over all).
BLINK_CONFIGS = [
    "Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence",
    "IQ_Test", "Jigsaw", "Multi_view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence",
    "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity",
]

_IMG_SUBDIR = {
    "mmstar": "MMStar_mmeval_images", "realworldqa": "RealWorldQA_images",
    "logicvista": "LogicVista_images", "mmbench": "MMBench_images",
    "mmbench-v11": "MMBench_V11_images", "scienceqa-img": "ScienceQA_IMG_images",
    "blink": "BLINK_images",
}


def _norm_answer(ans):
    """'(B)' -> 'B'; bare letter uppercased; non-letter (open) answers kept as-is."""
    s = str(ans if ans is not None else "").strip()
    m = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?[.\s]*", s)
    return m.group(1).upper() if m else s


def _messages(sample):
    msgs = sample.get("messages")
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    return msgs or []


def _first_user(msgs):
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            return m
    return msgs[0] if msgs else {}


def _save_media(media_list, img_dir, stem):
    """Save each PIL/bytes/path media item as PNG (lossless, no recompression);
    return list of absolute file paths in order."""
    from io import BytesIO

    from PIL import Image
    paths = []
    for k, media in enumerate(media_list):
        img = media
        if isinstance(img, dict) and img.get("bytes") is not None:
            img = Image.open(BytesIO(img["bytes"]))
        elif isinstance(img, str):
            img = Image.open(img)
        if not isinstance(img, Image.Image):
            raise ValueError(f"Unsupported media type {type(media)} for {stem}")
        img = img.convert("RGB")
        suffix = "" if len(media_list) == 1 else f"_{k}"
        p = img_dir / f"{stem}{suffix}.png"
        if not p.exists():
            img.save(p, format="PNG")
        paths.append(str(p))
    return paths


def _row_from_sample(sample, idx, img_dir, category_keys, category_const=None):
    msgs = _messages(sample)
    m = _first_user(msgs)
    media = sample.get("media")
    media_list = media if isinstance(media, list) else ([] if media is None else [media])
    qid = sample.get("id", idx)
    img_paths = _save_media(media_list, img_dir, str(qid))
    options = m.get("options") if isinstance(m.get("options"), dict) else {}
    query = _TMPL.render(question=(m.get("question") or "").strip(), options=options)
    if category_const is not None:
        category = category_const
    else:
        category = "unknown"
        for ck in category_keys:
            v = m.get(ck) or sample.get(ck)
            if v:
                category = str(v)
                break
    return {
        "images": img_paths,
        "query": query,
        "response": _norm_answer(m.get("answer")),
        "question_id": qid,
        "category": category,
    }


def prepare_mmeval_hf(out_dir, benchmark):
    from datasets import get_dataset_config_names, load_dataset
    out_dir = Path(out_dir)
    if benchmark == "blink":
        return _prepare_blink(out_dir)
    if benchmark not in MMEVAL_HF_SPECS:
        raise ValueError(f"Unknown mm-eval benchmark: {benchmark}")
    repo, cfg, split = MMEVAL_HF_SPECS[benchmark]
    img_dir = out_dir / _IMG_SUBDIR[benchmark]
    img_dir.mkdir(parents=True, exist_ok=True)
    cfgs = get_dataset_config_names(repo)
    use = cfg if cfg in cfgs else ("default" if "default" in cfgs
                                   else [c for c in cfgs if not c.endswith("_metadata")][0])
    ds = load_dataset(repo, name=use, split=split, verification_mode="no_checks")
    cat_keys = ["category", "l2_category", "L2_category", "skill", "task", "sub_task"]
    data = []
    for i, sample in enumerate(tqdm(ds, desc=f"Processing {benchmark}", unit="ex")):
        data.append(_row_from_sample(dict(sample), i, img_dir, cat_keys))
    return data


def _prepare_blink(out_dir):
    from datasets import load_dataset
    img_dir = out_dir / _IMG_SUBDIR["blink"]
    img_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for cfg in BLINK_CONFIGS:
        ds = load_dataset("mm-eval/BLINK", name=cfg, split="val", verification_mode="no_checks")
        for i, sample in enumerate(tqdm(ds, desc=f"BLINK/{cfg}", unit="ex")):
            sample = dict(sample)
            # namespace question_id by sub-task to keep uids unique across configs
            sample["id"] = f"{cfg}_{sample.get('id', i)}"
            row = _row_from_sample(sample, i, img_dir, [], category_const=cfg)
            data.append(row)
    return data

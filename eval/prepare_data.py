import argparse
import base64
import os
import json
import sys
from pathlib import Path

from tqdm import tqdm

from huggingface_hub import snapshot_download

BENCHMARK_JSON_MAP = {
    "zoombench": "zoombench.json",
    "vstar": "vstar.json",
    "hrbench-4k": "hr_bench_4k.json",
    "hrbench-8k": "hr_bench_8k.json",
    "mme-realworld": "MME_RealWorld.json",
    "mme-realworld-cn": "MME_RealWorld_CN.json",
    "mme-realworld-lite": "MME_RealWorld_Lite.json",
    "mmstar": "mmstar.json",
    # mm-eval/* HF benchmarks (simple-mmeval image source, VLMEvalKit-aligned prompt)
    "realworldqa": "realworldqa.json",
    "logicvista": "logicvista.json",
    "mmbench": "mmbench.json",
    "mmbench-v11": "mmbench_v11.json",
    "scienceqa-img": "scienceqa_img.json",
    "blink": "blink.json",
    "pope": "POPE.json",
    "pope_adv": "POPE_adv.json",
    "pope_pop": "POPE_pop.json",
    "pope_random": "POPE_random.json",
    "cv-bench": "cv_bench.json",
    "mmvp": "mmvp.json",
    "visualprobe": "visualprobe.json",
    # math benchmarks (LLM-judge scored, VLMEvalKit-aligned prompts)
    "mathvista": "mathvista.json",
    "mathverse": "mathverse.json",
    "mathvision": "mathvision.json",
    "mathvision-op": "mathvision_op.json",  # official-prompt variant, json derived from mathvision.json
    "wemath": "wemath.json",
    "hallusionbench": "hallusionbench.json",
}


def resolve_benchmark_json(benchmark):
    if benchmark not in BENCHMARK_JSON_MAP:
        print(f"ERROR: Unsupported benchmark: {benchmark}", file=sys.stderr)
        print(f"Supported: {', '.join(BENCHMARK_JSON_MAP.keys())}", file=sys.stderr)
        sys.exit(1)
    return BENCHMARK_JSON_MAP[benchmark]


def prepare_zoombench(out_dir):
    import pyarrow.parquet as pq

    local_dir = out_dir / "ZoomBench_data"
    snapshot_download("inclusionAI/ZoomBench", repo_type="dataset", local_dir=str(local_dir))

    src = local_dir / "data" / "test.parquet"
    full_dir = out_dir / "ZoomBench_images"
    crop_dir = out_dir / "ZoomBench_crop_images"
    full_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(src), columns=["id", "query", "response", "image", "crop_image"])
    rows = table.to_pylist()

    data = []
    for i, row in enumerate(rows):
        sid = str(row.get("id") or i)
        full_obj = row.get("image") or {}
        full_bytes = full_obj.get("bytes") if isinstance(full_obj, dict) else None
        if not isinstance(full_bytes, (bytes, bytearray)):
            raise ValueError(f"Invalid full image bytes at row {i}")

        crop_obj = row.get("crop_image") or {}
        crop_bytes = crop_obj.get("bytes") if isinstance(crop_obj, dict) else None

        full_path = full_dir / f"{sid}.jpg"
        with open(full_path, "wb") as f:
            f.write(full_bytes)

        crop_list = []
        if isinstance(crop_bytes, (bytes, bytearray)):
            crop_path = crop_dir / f"{sid}.jpg"
            with open(crop_path, "wb") as f:
                f.write(crop_bytes)
            crop_list = [str(crop_path)]

        data.append({
            "images": [str(full_path)],
            "crop_images": crop_list,
            "query": (row.get("query") or "").strip(),
            "response": (row.get("response") or "").strip(),
        })
    return data


def prepare_vstar(out_dir):
    import pyarrow.parquet as pq

    local_dir = out_dir / "vstar_data"
    snapshot_download("lmms-lab/vstar-bench", repo_type="dataset", local_dir=str(local_dir))

    src = local_dir / "data" / "test-00000-of-00001.parquet"
    img_dir = out_dir / "VStar_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(src), columns=["image", "text", "label", "question_id", "category"])
    rows = table.to_pylist()

    data = []
    for i, row in enumerate(rows):
        qid = row.get("question_id")
        if qid is None or str(qid).strip() == "":
            qid = i
        image_obj = row.get("image") or {}
        img_bytes = image_obj.get("bytes") if isinstance(image_obj, dict) else None
        if not isinstance(img_bytes, (bytes, bytearray)):
            raise ValueError(f"Invalid image bytes at row {i}, question_id={qid}")

        img_path = img_dir / f"{str(qid)}.jpg"
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        query_raw = (row.get("text") or "").strip()
        query = query_raw
        post_prompt = "\nAnswer with the option's letter from the given choices directly."
        if not query.endswith(post_prompt.strip()):
            query += post_prompt
        response = (row.get("label") or "").strip().upper()
        data.append({
            "images": [str(img_path)],
            "query": query,
            "response": response,
            "question_id": qid,
            "category": row.get("category") or "unknown",
        })
    return data


def prepare_hrbench(out_dir, benchmark):
    import pyarrow.parquet as pq

    local_dir = out_dir / "HR-Bench_data"
    snapshot_download("DreamMr/HR-Bench", repo_type="dataset", local_dir=str(local_dir))

    parquet_name = "hr_bench_4k.parquet" if benchmark == "hrbench-4k" else "hr_bench_8k.parquet"
    src = local_dir / parquet_name
    img_dir = out_dir / ("HRBench_4k_images" if benchmark == "hrbench-4k" else "HRBench_8k_images")
    img_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(src), columns=["index", "question", "answer", "A", "B", "C", "D", "category", "image"])
    rows = table.to_pylist()

    data = []
    for i, row in enumerate(rows):
        idx = int(row.get("index", i))
        img_path = img_dir / f"{idx:05d}.jpg"
        img_b64 = row.get("image") or ""
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img_b64))
        question = (row.get("question") or "").strip()
        options = []
        for letter in ["A", "B", "C", "D"]:
            opt = (row.get(letter) or "").strip()
            if opt:
                options.append(f"({letter}) {opt}")
        if options:
            query = question + " Select from the following choices.\n" + "\n".join(options)
        else:
            query = question
        data.append({
            "index": idx,
            "question_id": idx,
            "images": [str(img_path)],
            "query": query,
            "response": (row.get("answer") or "").strip(),
            "category": (row.get("category") or "").strip(),
        })
    return data


def _load_mme_realworld_parquet(repo_id, out_dir, img_subdir):
    import pyarrow.dataset as ds

    local_dir = out_dir / repo_id.replace("/", "_")
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(local_dir))

    data_dir = local_dir / "data"
    img_dir = out_dir / img_subdir
    img_dir.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(str(data_dir), format="parquet")
    table = dataset.to_table(columns=["bytes", "index", "question", "multi-choice options", "answer"])
    return table.to_pylist(), img_dir


def _build_mme_realworld_query(question, option_lines, lang="en"):
    if lang == "cn":
        return (
            question
            + " 选项如下所示:\n"
            + "\n".join(option_lines)
            + "\n根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n"
            + "最佳答案为： "
        ) if option_lines else question
    return (
        question
        + " The choices are listed below:\n"
        + "\n".join(option_lines)
        + "\nSelect the best answer to the above multiple-choice question based on the image. "
        + "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
        + "The best answer is: "
    ) if option_lines else question


def _process_mme_realworld_rows(rows, img_dir, lang="en"):
    data = []
    for row in tqdm(rows, desc="Processing images", unit="img"):
        idx = int(row.get("index", 0))
        b64 = row.get("bytes") or ""
        img_path = img_dir / f"{idx:06d}.jpg"
        if not img_path.exists():
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(b64))

        question = (row.get("question") or "").strip()
        options = row.get("multi-choice options") or []
        option_lines = [str(x).strip() for x in options if str(x).strip()]
        query = _build_mme_realworld_query(question, option_lines, lang)
        answer = (row.get("answer") or "").strip()

        data.append({
            "index": idx,
            "question_id": idx,
            "images": [str(img_path)],
            "query": query,
            "response": answer,
        })
    return data


def prepare_mmstar(out_dir):
    import pyarrow.parquet as pq

    local_dir = out_dir / "MMStar_data"
    snapshot_download("Lin-Chen/MMStar", repo_type="dataset", local_dir=str(local_dir))

    src = local_dir / "mmstar.parquet"
    img_dir = out_dir / "MMStar_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(src), columns=["index", "question", "answer", "category", "l2_category", "image"])
    rows = table.to_pylist()

    data = []
    for row in tqdm(rows, desc="Processing MMStar images", unit="img"):
        idx = int(row.get("index", 0))
        img_bytes = row.get("image")
        if not isinstance(img_bytes, (bytes, bytearray)):
            raise ValueError(f"Invalid image bytes at index {idx}")

        img_path = img_dir / f"{idx:05d}.jpg"
        if not img_path.exists():
            with open(img_path, "wb") as f:
                f.write(img_bytes)

        query = (row.get("question") or "").strip()

        data.append({
            "index": idx,
            "question_id": idx,
            "images": [str(img_path)],
            "query": query,
            "response": (row.get("answer") or "").strip().upper(),
            "category": row.get("category") or "unknown",
            "l2_category": row.get("l2_category") or "unknown",
        })
    return data


def prepare_pope(out_dir, benchmark):
    from datasets import load_dataset

    local_dir = out_dir / "POPE_data"
    snapshot_download("lmms-lab/POPE", repo_type="dataset", local_dir=str(local_dir))

    img_dir = out_dir / "POPE_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"pope": "test", "pope_adv": "adversarial", "pope_pop": "popular", "pope_random": "random"}
    split = split_map[benchmark]

    if benchmark == "pope":
        ds = load_dataset(str(local_dir), split=split)
    else:
        ds = load_dataset(str(local_dir), name="Full", split=split)

    post_prompt = "\nAnswer the question using a single word or phrase."
    data = []
    for i, row in enumerate(tqdm(ds, desc=f"Processing POPE ({split})", unit="img")):
        rid = str(row.get("id") or i)
        qid = str(row.get("question_id") or i)
        category = row.get("category") or split
        image_source = row.get("image_source") or rid
        img = row.get("image")

        img_name = f"{category}_{i:05d}_{image_source}.jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            img.save(str(img_path))

        question = (row.get("question") or "").strip()
        query = question + post_prompt if not question.endswith(post_prompt.strip()) else question

        data.append({
            "index": i,
            "id": rid,
            "question_id": qid,
            "images": [str(img_path)],
            "query": query,
            "response": (row.get("answer") or "").strip().lower(),
            "category": category,
            "image_source": image_source,
        })
    return data


def prepare_cvbench(out_dir):
    import pyarrow.parquet as pq

    local_dir = out_dir / "CVBench_data"
    snapshot_download("nyu-visionx/CV-Bench", repo_type="dataset", local_dir=str(local_dir))

    data = []
    for dim, parquet_name, img_subdir in [
        ("2D", "test_2d.parquet", "CVBench_2d_images"),
        ("3D", "test_3d.parquet", "CVBench_3d_images"),
    ]:
        src = local_dir / parquet_name
        img_dir = out_dir / img_subdir
        img_dir.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(str(src), columns=["idx", "type", "task", "image", "prompt", "answer", "source"])
        rows = table.to_pylist()

        for row in tqdm(rows, desc=f"Processing CV-Bench {dim}", unit="img"):
            idx = int(row.get("idx", 0))
            img_obj = row.get("image") or {}
            img_bytes = img_obj.get("bytes") if isinstance(img_obj, dict) else None
            if not isinstance(img_bytes, (bytes, bytearray)):
                raise ValueError(f"Invalid image bytes at idx {idx}")

            img_path = img_dir / f"{idx:05d}.jpg"
            if not img_path.exists():
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

            data.append({
                "index": idx,
                "question_id": idx,
                "images": [str(img_path)],
                "query": (row.get("prompt") or "").strip(),
                "response": (row.get("answer") or "").strip(),
                "type": row.get("type") or dim,
                "task": row.get("task") or "",
                "source": row.get("source") or "",
            })
    return data


def prepare_mmvp(out_dir):
    import csv

    local_dir = out_dir / "MMVP_data"
    snapshot_download("MMVP/MMVP", repo_type="dataset", local_dir=str(local_dir))

    csv_path = local_dir / "Questions.csv"
    img_src_dir = local_dir / "MMVP Images"

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["Index"])
            question = row["Question"].strip()
            options = row["Options"].strip()
            answer = row["Correct Answer"].strip()

            img_path = img_src_dir / f"{qid}.jpg"
            if not img_path.exists():
                raise ValueError(f"Image not found: {img_path}")

            data.append({
                "index": qid,
                "question_id": qid,
                "images": [str(img_path)],
                "query": f"{question} {options}",
                "response": answer,
            })
    return data


def prepare_visualprobe(out_dir):
    repos = [
        ("Easy", "Mini-o3/VisualProbe_Easy"),
        ("Medium", "Mini-o3/VisualProbe_Medium"),
        ("Hard", "Mini-o3/VisualProbe_Hard"),
    ]

    data = []
    for category, repo_id in repos:
        local_dir = out_dir / f"VisualProbe_{category}_data"
        snapshot_download(repo_id, repo_type="dataset", local_dir=str(local_dir))

        val_path = local_dir / "val.json"
        with open(val_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in tqdm(records, desc=f"Processing VisualProbe {category}", unit="img"):
            images = record.get("images") or []
            if not images:
                continue
            rel_path = images[0]
            prefix = f"VisualProbe_{category}/"
            if rel_path.startswith(prefix):
                img_path = local_dir / rel_path[len(prefix):]
            else:
                img_path = local_dir / rel_path

            problem = (record.get("problem") or "").strip()
            query = problem.replace("<image>\n", "").replace("<image>", "").strip()

            data.append({
                "images": [str(img_path)],
                "query": query,
                "response": (record.get("solution") or "").strip(),
                "category": category,
            })
    return data


def prepare_mme_realworld(out_dir):
    rows, img_dir = _load_mme_realworld_parquet(
        "yifanzhang114/MME-RealWorld-Lmms-eval", out_dir, "MME_RealWorld_Full_images",
    )
    return _process_mme_realworld_rows(rows, img_dir, lang="en")


def prepare_mme_realworld_cn(out_dir):
    rows, img_dir = _load_mme_realworld_parquet(
        "yifanzhang114/MME-RealWorld-CN-Lmms-eval", out_dir, "MME_RealWorld_CN_images",
    )
    return _process_mme_realworld_rows(rows, img_dir, lang="cn")


def prepare_mme_realworld_lite(out_dir):
    rows, img_dir = _load_mme_realworld_parquet(
        "yifanzhang114/MME-RealWorld-lite-lmms-eval", out_dir, "MME_RealWorld_Lite_images",
    )
    return _process_mme_realworld_rows(rows, img_dir, lang="en")


def _save_pil(img, path):
    if img is None:
        return False
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if not path.exists():
        img.save(path)
    return True


def prepare_mathvista(out_dir):
    from datasets import load_dataset

    img_dir = out_dir / "MathVista_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("AI4Math/MathVista", split="testmini")

    data = []
    for i, row in enumerate(tqdm(ds, desc="Processing MathVista", unit="img")):
        pid = str(row.get("pid") or i)
        img_path = img_dir / f"{pid}.png"
        if not _save_pil(row.get("decoded_image"), img_path):
            continue

        # The dataset ships a VLMEvalKit-aligned query with hint + lettered
        # choices for multi_choice and answer-format hints for free_form.
        query = (row.get("query") or row.get("question") or "").strip()

        answer = (row.get("answer") or "").strip()
        choices = row.get("choices") or []
        if row.get("question_type") == "multi_choice" and answer in list(choices):
            letter = chr(ord("A") + list(choices).index(answer))
            response = f"{letter}. {answer}"
        else:
            response = answer

        data.append({
            "index": i,
            "question_id": pid,
            "images": [str(img_path)],
            "query": query,
            "response": response,
            "type": row.get("question_type") or "free_form",
            "task": (row.get("metadata") or {}).get("task") or "",
        })
    return data


def prepare_mathverse(out_dir):
    from datasets import load_dataset

    img_dir = out_dir / "MathVerse_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")

    data = []
    for i, row in enumerate(tqdm(ds, desc="Processing MathVerse", unit="img")):
        version = (row.get("problem_version") or "").strip()
        if version.lower() == "text only":
            continue  # no image; incompatible with the VLM inference path

        sid = str(row.get("sample_index") or i)
        img_path = img_dir / f"{sid}.png"
        if not _save_pil(row.get("image"), img_path):
            continue

        # query_wo (direct-answer) is what VLMEvalKit uses for MathVerse by
        # default; we match it for cross-framework alignment. (The CoT variant
        # scores ~25 points higher on small models — protocol, not ability.)
        query = (row.get("query_wo") or row.get("query_cot") or row.get("question") or "").strip()

        data.append({
            "index": i,
            "question_id": sid,
            "images": [str(img_path)],
            "query": query,
            "response": (row.get("answer") or "").strip(),
            "type": row.get("question_type") or "free-form",
            "task": version,
        })
    return data


def prepare_mathvision(out_dir):
    from datasets import load_dataset

    img_dir = out_dir / "MathVision_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("MathLLMs/MathVision", split="test")

    data = []
    for i, row in enumerate(tqdm(ds, desc="Processing MathVision", unit="img")):
        sid = str(row.get("id") or i)
        img_path = img_dir / f"{sid}.png"
        if not _save_pil(row.get("decoded_image"), img_path):
            continue

        question = (row.get("question") or "").strip()
        options = [o for o in (row.get("options") or []) if str(o).strip()]
        if options:
            lines = [f"({chr(ord('A') + j)}) {o}" for j, o in enumerate(options)]
            query = (
                f"{question}\nChoices:\n" + "\n".join(lines) +
                "\nSolve the problem step by step, then answer with the option letter from the given choices."
            )
        else:
            query = (
                f"{question}\nSolve the problem step by step, then answer the question "
                "using a single word, number, or expression."
            )

        data.append({
            "index": i,
            "question_id": sid,
            "images": [str(img_path)],
            "query": query,
            "response": (row.get("answer") or "").strip(),
            "type": "multi_choice" if options else "free_form",
            "task": row.get("subject") or "",
        })
    return data


def prepare_wemath(out_dir):
    """Build wemath.json from the VLMEvalKit WeMath TSV when available (exact
    prompt parity with their ImageMCQ build_prompt: Hint/Question/multiline
    Options/'Please select...'), falling back to the HF release otherwise."""
    import base64
    import string as _string

    import pandas as pd

    img_dir = out_dir / "WeMath_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    tsv = Path(os.environ.get("LMUData", str(Path.home() / "LMUData"))) / "WeMath.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"{tsv} not found — download via VLMEvalKit first (protocol parity source)")

    df = pd.read_csv(tsv, sep="\t")
    data = []
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Processing We-Math", unit="img")):
        row = row._asdict()
        sid = str(row.get("index") if row.get("index") is not None else i)
        img_path = img_dir / f"{sid}.png"
        if not img_path.exists():
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(row["image"]))

        options = {c: row[c] for c in _string.ascii_uppercase
                   if c in row and not pd.isna(row[c])}
        options_prompt = "Options:\n" + "".join(f"{k}. {v}\n" for k, v in options.items())
        hint = row.get("hint")
        prompt = ""
        if hint is not None and not (isinstance(hint, float) and pd.isna(hint)):
            prompt += f"Hint: {hint}\n"
        prompt += f"Question: {row['question']}\n"
        if options:
            prompt += options_prompt + "Please select the correct answer from the options above. \n"

        data.append({
            "index": i,
            "question_id": sid,
            "images": [str(img_path)],
            "query": prompt,
            "response": str(row.get("answer") or "").strip().upper(),
            "type": "multi_choice",
            "task": str(row.get("knowledge concept") or ""),
        })
    return data


def prepare_hallusionbench(out_dir):
    from datasets import load_dataset

    img_dir = out_dir / "HallusionBench_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("lmms-lab/HallusionBench", split="image")

    data = []
    for i, row in enumerate(tqdm(ds, desc="Processing HallusionBench", unit="img")):
        img = row.get("image")
        # Text-only control questions (visual_input == "0") have no image; the
        # inference path requires one, so keep only visual questions (VD/VS).
        if img is None:
            continue
        sid = f"{row.get('set_id')}_{row.get('figure_id')}_{row.get('question_id')}"
        img_path = img_dir / f"{i:05d}.png"
        if not _save_pil(img, img_path):
            continue

        question = (row.get("question") or "").strip()
        gt = "Yes" if str(row.get("gt_answer")).strip() == "1" else "No"

        data.append({
            "index": i,
            "question_id": sid,
            "images": [str(img_path)],
            "query": f"{question}\nAnswer with Yes or No.",
            "response": gt,
            "type": "yes_no",
            "task": f"{row.get('category')}/{row.get('subcategory')}",
            "set_id": str(row.get("set_id")),
            "figure_id": str(row.get("figure_id")),
        })
    return data


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark data from HuggingFace")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="Output directory (default: script dir)")
    args = parser.parse_args()

    benchmark = args.benchmark
    benchmark_json = resolve_benchmark_json(benchmark)
    out_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent
    out_json = out_dir / benchmark_json

    if out_json.exists():
        print(f"Already exists: {out_json}, skipping.")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print(f"Preparing {benchmark} -> {out_json} ...")

    if benchmark == "zoombench":
        data = prepare_zoombench(out_dir)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark == "vstar":
        data = prepare_vstar(out_dir)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark in ("hrbench-4k", "hrbench-8k"):
        data = prepare_hrbench(out_dir, benchmark)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark in ("mmstar", "realworldqa", "logicvista", "mmbench",
                        "mmbench-v11", "scienceqa-img", "blink"):
        from mmeval_bench import prepare_mmeval_hf
        data = prepare_mmeval_hf(out_dir, benchmark)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark in ("pope", "pope_adv", "pope_pop", "pope_random"):
        data = prepare_pope(out_dir, benchmark)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark == "cv-bench":
        data = prepare_cvbench(out_dir)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark == "mmvp":
        data = prepare_mmvp(out_dir)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark in ("mathvista", "mathverse", "mathvision", "wemath", "hallusionbench"):
        prepare_fn = {"mathvista": prepare_mathvista,
                      "mathverse": prepare_mathverse,
                      "mathvision": prepare_mathvision,
                      "wemath": prepare_wemath,
                      "hallusionbench": prepare_hallusionbench}[benchmark]
        data = prepare_fn(out_dir)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark == "visualprobe":
        data = prepare_visualprobe(out_dir)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark == "mme-realworld":
        data = prepare_mme_realworld(out_dir)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark == "mme-realworld-cn":
        data = prepare_mme_realworld_cn(out_dir)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")

    elif benchmark == "mme-realworld-lite":
        data = prepare_mme_realworld_lite(out_dir)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Generated: {out_json} (records={len(data)})")


if __name__ == "__main__":
    main()

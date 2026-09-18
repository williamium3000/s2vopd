#!/usr/bin/env bash
# Run infer -> judge -> cal_acc for the 7 mm-eval benchmarks against a served policy,
# using the local Qwen2.5-72B judge. Skips prepare (JSONs already built). Appends the
# final accuracy of each benchmark to a summary file.
#
# Usage: MODEL_ID=Qwen3.5-4B MODEL_TAG=Qwen3.5-4B API_BASE=http://localhost:8010/v1/ \
#        bash orchestrate_mmeval.sh
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL_ID="${MODEL_ID:?set MODEL_ID (served-model-name)}"
MODEL_TAG="${MODEL_TAG:-$MODEL_ID}"
API_BASE="${API_BASE:?set API_BASE}"
JUDGE_BASE="${JUDGE_BASE:-http://localhost:8001/v1/}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen2.5-72B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
WORKERS="${WORKERS:-96}"
BENCHES="${BENCHES:-realworldqa mmstar logicvista scienceqa-img blink mmbench mmbench-v11}"
SUMMARY="summary_mmeval_${MODEL_TAG}.txt"
: > "$SUMMARY"

declare -A JSON=( [mmstar]=mmstar.json [realworldqa]=realworldqa.json [logicvista]=logicvista.json
  [mmbench]=mmbench.json [mmbench-v11]=mmbench_v11.json [scienceqa-img]=scienceqa_img.json [blink]=blink.json )

tag="${MODEL_TAG}_seed42"
for b in $BENCHES; do
  echo "############ [$(date '+%T')] $b ($MODEL_TAG) ############"
  python3 infer.py --benchmark "$b" --benchmark_json "${JSON[$b]}" \
    --out_dir model_answer --model_name "$tag" --seed 42 \
    --api_base "$API_BASE" --api_key EMPTY --model_id "$MODEL_ID" \
    --max_tokens "$MAX_TOKENS" --max_retries 2 --parallel_workers "$WORKERS" \
    --enable_thinking False 2>&1 | grep -aE "Inference done|Remaining|All samples" | tail -2
  python3 judge_qwenlm.py --benchmark "$b" --model "$tag" \
    --api_base "$JUDGE_BASE" --api_key EMPTY --judge_model "$JUDGE_MODEL" --judge_max_tokens 256 2>&1 \
    | grep -aE "LLM used|Total:" | tail -1
  acc=$(python3 cal_acc.py --benchmark "$b" --judge_json "judge/$b/${tag}_answer.jsonl" \
    --benchmark_json "${JSON[$b]}" 2>/dev/null | grep -aE "Acc:" | tail -1)
  echo "RESULT  $b  ->  $acc" | tee -a "$SUMMARY"
done
echo "===== DONE $MODEL_TAG =====" | tee -a "$SUMMARY"
cat "$SUMMARY"

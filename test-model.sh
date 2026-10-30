set -euo pipefail
OUT="$PWD/results-27b"; mkdir -p "$OUT"
MODEL="$(curl -sf http://localhost:8000/v1/models | jq -r '.data[0].id')"
[ -n "$MODEL" ] || { echo "Server non reachable"; exit 1; }

RUN_SEED="$(date +%s)"   # new at every script run
SUFFIX=1024              # "new" token per turn in warm case

bench() {  # $1=file-name  $2=prefix_len  $3=input_len  $4=n_requests  $5=seed
  docker run --rm --network host \
    -v "$OUT":/results-27b \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    --entrypoint vllm \
    vllm/vllm-openai:v0.30.0 \
    bench serve \
      --backend vllm \
      --endpoint /v1/completions \
      --base-url http://localhost:8000 \
      --model "$MODEL" \
      --tokenizer gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
      --dataset-name random \
      --random-prefix-len "$2" \
      --random-input-len "$3" \
      --random-output-len 256 \
      --random-range-ratio 0 \
      --seed "$5" \
      --ignore-eos \
      --num-prompts "$4" \
      --request-rate inf \
      --max-concurrency 1 \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result \
      --result-filename "/results-27b/$1.json" \
      --ready-check-timeout-sec 600
}

# warmup una tantum (CUDA graph ecc.)
bench warmup_tmp 0 512 3 "$((RUN_SEED - 1))" >/dev/null

i=0
for ctx in 1024 2048 4096 8192 16384 32768 65536 131072 163840 204800 230400; do
  i=$((i+1))
  n=11; [ "$ctx" -ge 65536 ] && n=5     # one more: the 1st is drop

  echo "=== COLD $ctx ==="
  bench "cold_${ctx}" 0 "$ctx" "$n" "$((RUN_SEED + i*1000))"

  if [ "$ctx" -gt "$SUFFIX" ]; then
    echo "=== WARM $ctx (prefisso $((ctx-SUFFIX)) + suffisso $SUFFIX) ==="
    bench "warm_${ctx}" "$((ctx-SUFFIX))" "$SUFFIX" "$n" "$((RUN_SEED + i*1000 + 500))"
  fi
done

rm -f "$OUT/warmup_tmp.json"

# summary: TTFT (ms) without the 1st request
printf "%-6s %8s %12s %12s %12s\n" kind ctx ttft_mean ttft_med tpot_ms
for f in "$OUT"/cold_*.json "$OUT"/warm_*.json; do
  b=$(basename "$f" .json)
  jq -r --arg k "${b%%_*}" --arg c "${b##*_}" '
    (.ttfts[1:] | sort) as $t
    | [$k, $c,
       (($t | add / length) * 1000 | floor),
       ($t[($t|length/2|floor)] * 1000 | floor),
       (.mean_tpot_ms | . * 100 | floor / 100)]
    | @tsv' "$f"
done | column -t

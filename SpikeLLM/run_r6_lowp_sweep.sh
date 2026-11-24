#!/bin/bash
# Round 6 · low_p sensitivity sweep on LLaMA-68M (W4A4, --let)
# Answers: is component A a mechanism, or a patch tuned for low_p=0.9?
# 5 levels x {orig, v2, a-off} = 15 runs, plus zero-padding control at 0.9.
set -u
cd "D:/spikellm/SpikeLLM" || exit 1
PY="D:/spikellm/.venv/Scripts/python.exe"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--model JackFram/llama-68m --epochs 20 --wbits 4 --abits 4 --lwc --aug_loss \
--let --let_lr 1e-3 --alpha 0.75 --act-scales ./act_scales/llama-68m.pt \
--act-shifts ./act_shifts/llama-68m.pt --seed 2 --batch_size 1 --addbit 1 --eval_ppl"

run_one () {
  local lp="$1"; local tag="$2"; local extra="$3"; local out="./log/$2"
  if [ -f "$out/done.flag" ]; then echo "SKIP $2"; return 0; fi
  mkdir -p "$out"
  echo "=== START $2 (low_p=$lp) $(date +%H:%M:%S) ==="
  $PY main.py $COMMON --low_p "$lp" $extra --output_dir "$out" > "$out/stdout.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$out/done.flag"; fi
  echo "=== END $2 exit=$rc $(date +%H:%M:%S) ==="
}

for lp in 0.80 0.85 0.90 0.95 0.98; do
  tag=$(echo "$lp" | tr -d '.')
  run_one "$lp" "r6_lp${tag}_orig" "--orig_spikellm"
  run_one "$lp" "r6_lp${tag}_v2"   ""
  run_one "$lp" "r6_lp${tag}_aoff" "--ablate_no_linear_shield"
done

# controlled comparison: zero-padding (v1) vs mean-anchored (v2) at low_p=0.9
run_one 0.90 "r6_lp90_zero" "--shield_mode zero"

echo "=== SWEEP COMPLETE $(date) ==="

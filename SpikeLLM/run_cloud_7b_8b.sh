#!/usr/bin/env bash
# ==============================================================================
# OSDSQ v2 Benchmark Suite for Dual 24GB RTX 3090 Lab Server / Cloud A100
# Models: LLaMA-2-7B (MHA) & LLaMA-3-8B (GQA)
# Usage:
#   bash run_cloud_7b_8b.sh smoke       # 3-min pre-check on LLaMA-2-7B
#   bash run_cloud_7b_8b.sh full        # Sequential full suite (--multigpu across 2x 3090)
#   bash run_cloud_7b_8b.sh parallel    # Dual-3090 parallel mode (GPU 0 & GPU 1 concurrently)
# ==============================================================================
set -u

MODE=${1:-"full"}
TASKS="piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande"
SEED=2
EPOCHS=20
ALPHA=0.5
LET_LR=5e-3
LWC_LR=1e-2

mkdir -p ./log ./act_scales ./act_shifts ./cache

ensure_act_stats() {
    local MODEL_PATH=$1
    local NET_NAME
    NET_NAME=$(basename "${MODEL_PATH}")
    if [ ! -f "./act_scales/${NET_NAME}.pt" ] || [ ! -f "./act_shifts/${NET_NAME}.pt" ]; then
        echo "[Pre-check] Generating act_scales and act_shifts for ${NET_NAME} ..."
        python generate_act_scale_shift.py \
            --model "${MODEL_PATH}" \
            --scales-output-path ./act_scales/ \
            --shifts-output-path ./act_shifts/ \
            --calib_dataset wikitext2 \
            --num-samples 128 \
            --seq-len 2048 \
            --seed ${SEED}
    else
        echo "[Pre-check] Found existing ./act_scales/${NET_NAME}.pt and ./act_shifts/${NET_NAME}.pt"
    fi
}

run_exp() {
    local EXP_NAME=$1
    local OUT_DIR=$2
    shift 2
    if [ -f "${OUT_DIR}/done.flag" ]; then
        echo "[SKIP] ${EXP_NAME} already completed (${OUT_DIR}/done.flag exists)."
        return 0
    fi
    mkdir -p "${OUT_DIR}"
    echo ">>> [START] ${EXP_NAME} -> ${OUT_DIR}"
    if python main.py "$@" --output_dir "${OUT_DIR}" > "${OUT_DIR}/stdout.log" 2>&1; then
        touch "${OUT_DIR}/done.flag"
        echo "<<< [DONE]  ${EXP_NAME}"
    else
        echo "!!! [ERROR] ${EXP_NAME} failed! Check ${OUT_DIR}/stdout.log"
        return 1
    fi
}

if [ "${MODE}" = "smoke" ]; then
    MODEL_PATH=${2:-"meta-llama/Llama-2-7b-hf"}
    echo "=================================================================="
    echo " Running 3-Minute Smoke Test on ${MODEL_PATH} (Dual 3090 Pre-check)"
    echo "=================================================================="
    ensure_act_stats "${MODEL_PATH}"
    python main.py \
        --model "${MODEL_PATH}" \
        --epochs 1 --nsamples 8 --wbits 4 --abits 4 \
        --let --lwc --aug_loss --alpha ${ALPHA} --let_lr ${LET_LR} --lwc_lr ${LWC_LR} \
        --seed ${SEED} --batch_size 1 --addbit 1 --low_p 0.9 --multigpu \
        --eval_ppl --limit 4 --tasks "piqa" \
        --output_dir "./log/smoke_7b_osdsq_v2"
    python main.py \
        --model "${MODEL_PATH}" \
        --epochs 0 --nsamples 8 --resume "./log/smoke_7b_osdsq_v2/omni_parameters.pth" \
        --wbits 4 --abits 4 --let --lwc --aug_loss --alpha ${ALPHA} \
        --seed ${SEED} --batch_size 1 --addbit 1 --low_p 0.9 --multigpu \
        --sim_mode real_multibit_simulate \
        --eval_ppl --limit 4 \
        --output_dir "./log/smoke_7b_osdsq_v2_real_spike"
    echo "[SMOKE TEST PASSED] All pipelines (LET stats, train, resume, PPL, zero-shot, real spike) verified!"
    exit 0
fi

run_model_suite() {
    local MODEL_PATH=$1
    local TAG=$2

    echo "=================================================================="
    echo " Starting Full Benchmark Suite for ${TAG} (${MODEL_PATH})"
    echo "=================================================================="

    ensure_act_stats "${MODEL_PATH}"

    # 0. Layer-wise Range Hijack Ratio r and r^2
    if [ ! -f "./log/hijack_ratio_${TAG}.json" ]; then
        python measure_hijack_ratio.py \
            --model "${MODEL_PATH}" \
            --nsamples 16 \
            --output_prefix "./log/hijack_ratio_${TAG}" || true
    fi

    # 1. FP16 Baseline
    run_exp "${TAG}_fp16" "./log/${TAG}_fp16" \
        --model "${MODEL_PATH}" --epochs 0 --wbits 16 --abits 16 --multigpu \
        --eval_ppl --tasks "${TASKS}"

    # 2. External PTQ Baseline: Pure OmniQuant W4A4 (No Spike Split, No aug_loss)
    run_exp "${TAG}_omniquant_w4a4" "./log/${TAG}_omniquant_w4a4" \
        --model "${MODEL_PATH}" --epochs ${EPOCHS} --wbits 4 --abits 4 \
        --let --lwc --alpha ${ALPHA} --let_lr ${LET_LR} --lwc_lr ${LWC_LR} \
        --seed ${SEED} --batch_size 1 --addbit 0 --low_p 1.0 --orig_spikellm --multigpu \
        --eval_ppl --tasks "${TASKS}"

    if [ "${MODE}" = "parallel" ]; then
        echo ">>> Running Original vs. OSDSQ v2 concurrently on GPU 0 and GPU 1 ..."
        # Pair 1: No-LET (GPU 0: Original, GPU 1: OSDSQ v2)
        CUDA_VISIBLE_DEVICES=0 run_exp "${TAG}_orig_no_let" "./log/${TAG}_orig_no_let" \
            --model "${MODEL_PATH}" --epochs ${EPOCHS} --wbits 4 --abits 4 \
            --lwc --aug_loss --lwc_lr ${LWC_LR} --seed ${SEED} --batch_size 1 \
            --addbit 1 --low_p 0.9 --orig_spikellm --eval_ppl --tasks "${TASKS}" &
        PID0=$!
        CUDA_VISIBLE_DEVICES=1 run_exp "${TAG}_osdsq_v2_no_let" "./log/${TAG}_osdsq_v2_no_let" \
            --model "${MODEL_PATH}" --epochs ${EPOCHS} --wbits 4 --abits 4 \
            --lwc --aug_loss --lwc_lr ${LWC_LR} --seed ${SEED} --batch_size 1 \
            --addbit 1 --low_p 0.9 --eval_ppl --tasks "${TASKS}" &
        PID1=$!
        wait ${PID0} ${PID1}

        # Pair 2: With GQA-LET (GPU 0: Original, GPU 1: OSDSQ v2)
        CUDA_VISIBLE_DEVICES=0 run_exp "${TAG}_orig_with_let" "./log/${TAG}_orig_with_let" \
            --model "${MODEL_PATH}" --epochs ${EPOCHS} --wbits 4 --abits 4 \
            --let --lwc --aug_loss --alpha ${ALPHA} --let_lr ${LET_LR} --lwc_lr ${LWC_LR} \
            --seed ${SEED} --batch_size 1 --addbit 1 --low_p 0.9 --orig_spikellm \
            --eval_ppl --tasks "${TASKS}" &
        PID2=$!
        CUDA_VISIBLE_DEVICES=1 run_exp "${TAG}_osdsq_v2_with_let" "./log/${TAG}_osdsq_v2_with_let" \
            --model "${MODEL_PATH}" --epochs ${EPOCHS} --wbits 4 --abits 4 \
            --let --lwc --aug_loss --alpha ${ALPHA} --let_lr ${LET_LR} --lwc_lr ${LWC_LR} \
            --seed ${SEED} --batch_size 1 --addbit 1 --low_p 0.9 \
            --eval_ppl --tasks "${TASKS}" &
        PID3=$!
        wait ${PID2} ${PID3}
    else
        # Sequential mode with --multigpu evaluation across both 3090s
        run_exp "${TAG}_orig_no_let" "./log/${TAG}_orig_no_let" \
            --model "${MODEL_PATH}" --epochs ${EPOCHS} --wbits 4 --abits 4 \
            --lwc --aug_loss --lwc_lr ${LWC_LR} --seed ${SEED} --batch_size 1 \
            --addbit 1 --low_p 0.9 --orig_spikellm --multigpu --eval_ppl --tasks "${TASKS}"

        run_exp "${TAG}_orig_with_let" "./log/${TAG}_orig_with_let" \
            --model "${MODEL_PATH}" --epochs ${EPOCHS} --wbits 4 --abits 4 \
            --let --lwc --aug_loss --alpha ${ALPHA} --let_lr ${LET_LR} --lwc_lr ${LWC_LR} \
            --seed ${SEED} --batch_size 1 --addbit 1 --low_p 0.9 --orig_spikellm --multigpu \
            --eval_ppl --tasks "${TASKS}"

        run_exp "${TAG}_osdsq_v2_no_let" "./log/${TAG}_osdsq_v2_no_let" \
            --model "${MODEL_PATH}" --epochs ${EPOCHS} --wbits 4 --abits 4 \
            --lwc --aug_loss --lwc_lr ${LWC_LR} --seed ${SEED} --batch_size 1 \
            --addbit 1 --low_p 0.9 --multigpu --eval_ppl --tasks "${TASKS}"

        run_exp "${TAG}_osdsq_v2_with_let" "./log/${TAG}_osdsq_v2_with_let" \
            --model "${MODEL_PATH}" --epochs ${EPOCHS} --wbits 4 --abits 4 \
            --let --lwc --aug_loss --alpha ${ALPHA} --let_lr ${LET_LR} --lwc_lr ${LWC_LR} \
            --seed ${SEED} --batch_size 1 --addbit 1 --low_p 0.9 --multigpu \
            --eval_ppl --tasks "${TASKS}"
    fi

    # 7. Real Multi-Step Spike Simulation (real_multibit_simulate) via --resume --epochs 0
    run_exp "${TAG}_orig_let_real_spike" "./log/${TAG}_orig_let_real_spike" \
        --model "${MODEL_PATH}" --epochs 0 --resume "./log/${TAG}_orig_with_let/omni_parameters.pth" \
        --wbits 4 --abits 4 --let --lwc --aug_loss --alpha ${ALPHA} \
        --seed ${SEED} --batch_size 1 --addbit 1 --low_p 0.9 --orig_spikellm --multigpu \
        --sim_mode real_multibit_simulate --eval_ppl

    run_exp "${TAG}_osdsq_v2_let_real_spike" "./log/${TAG}_osdsq_v2_let_real_spike" \
        --model "${MODEL_PATH}" --epochs 0 --resume "./log/${TAG}_osdsq_v2_with_let/omni_parameters.pth" \
        --wbits 4 --abits 4 --let --lwc --aug_loss --alpha ${ALPHA} \
        --seed ${SEED} --batch_size 1 --addbit 1 --low_p 0.9 --multigpu \
        --sim_mode real_multibit_simulate --eval_ppl
}

# 1. Run LLaMA-2-7B (MHA: 32 Q heads / 32 KV heads)
run_model_suite "meta-llama/Llama-2-7b-hf" "llama2_7b"

# 2. Run LLaMA-3-8B (GQA: 32 Q heads / 8 KV heads)
run_model_suite "meta-llama/Meta-Llama-3-8B" "llama3_8b"

Write-Host "=== [1/2] Starting 1.1B Original SpikeLLM + GQA LET (1.1b_orig_with_gqa_let) ===" -ForegroundColor Cyan
d:\spikellm\.venv\Scripts\python.exe main.py `
    --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T `
    --epochs 20 --wbits 4 --abits 4 --lwc --aug_loss --let `
    --let_lr 1e-3 --alpha 0.75 `
    --act-scales ./act_scales/TinyLlama-1.1B-intermediate-step-1431k-3T.pt `
    --act-shifts ./act_shifts/TinyLlama-1.1B-intermediate-step-1431k-3T.pt `
    --seed 2 --batch_size 1 --addbit 1 --low_p 0.9 `
    --orig_spikellm --eval_ppl `
    --output_dir ./log/1.1b_orig_with_gqa_let

Write-Host "=== [2/2] Starting 1.1B OSDSQ v2 without LET (1.1b_osdsq_v2_no_let) ===" -ForegroundColor Green
d:\spikellm\.venv\Scripts\python.exe main.py `
    --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T `
    --epochs 20 --wbits 4 --abits 4 --lwc --aug_loss `
    --seed 2 --batch_size 1 --addbit 1 --low_p 0.9 `
    --eval_ppl `
    --output_dir ./log/1.1b_osdsq_v2_no_let

Write-Host "=== All 1.1B 2x2 Matrix Runs Completed! ===" -ForegroundColor Yellow

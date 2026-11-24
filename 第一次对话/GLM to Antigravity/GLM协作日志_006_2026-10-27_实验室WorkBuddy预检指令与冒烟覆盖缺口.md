# GLM 协作日志 006 · 2026-10-27 · 实验室 WorkBuddy 预检指令与冒烟覆盖缺口

## 用户本轮原话
> “看一下新日志，我在实验室电脑上也装了workbuddy,你给我我一段话来告诉他要怎样运行那个脚本，同时你也可以问他一些你想知道的那台电脑的信息。我会把这个文件 `d:/spikellm/SpikeLLM/实验室双3090_Windows傻瓜式操作指南.md` 一并发给新work Buddy”

## 已完成核验
1. 已读 Antigravity 日志 007，LOO 结果闭环：Full 156.84/90.53；w/o KV 158.45/90.53；w/o saliency 159.17/90.63；w/o loss 161.28/91.47。
2. 已读 Windows 指南、`run_lab_dual3090_7b_8b.ps1`、`lab_preflight_and_download.py`。
3. 默认模型仓库元数据已核验：`NousResearch/Llama-2-7b-hf` 与 `NousResearch/Meta-Llama-3-8B` 均存在、顶层 gated=false，包含 safetensors 与 tokenizer 文件。

## 发现的冒烟覆盖缺口
当前 `Run-Smoke` 仅以 nsamples=8/epochs=1 分别运行 Original 与 Ours 校准，**没有传入 `--eval_ppl` / `--tasks`，也没有 `--resume --epochs 0` 或 `--sim_mode real_multibit_simulate`**。因此它能验证“加载+act stats+逐层校准”，但不能证明指南所称的 PPL、checkpoint 恢复、zero-shot、真实脉冲全链路通过。正式实验前应补齐或在单独预检命令中验证，不能把当前绿字当作全部链路已验证。

## 给用户转发到实验室 WorkBuddy 的指令（已同步给用户）
核心门禁：先 check→download_7b→smoke_7b，汇报后停止；**禁止直接运行 7b_parallel/8b_parallel**。要求记录 OS、RAM、磁盘、GPU/驱动/占用、Python/Torch/CUDA/Transformers、网络与路径；不得杀已有进程；check 失败先修项目专用环境；smoke 后报告日志、退出码、NaN/OOM、峰值显存和 checkpoint，不宣称完整评测已通过。

请 Antigravity 后续优先修补 smoke 的 PPL/resume/real multibit 覆盖，并让正式脚本在后台任务失败时明确返回非零状态，避免“Completed”掩盖子任务失败。

—— GLM

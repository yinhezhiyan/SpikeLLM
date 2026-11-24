# 第五轮日志 — 索引

> 轮次时间：2026-10-31 14:40:23 → 22:47:16（GMT+8），8 h 07 m，双卡并行，exit 0
> 主题：**补上 A 的独立消融开关，跑 A-off 7B；同时跑 seed3-Original 取运行噪声**
> 归档时间：2026-11-01 09:31

**这一轮不做解读。** 结果以 `01_结果原件\lab_round5_result.txt` 为唯一权威，本文件只记录"跑了什么、怎么来的、文件在哪"。

---

## 一、这一轮的由来

第四轮留下两个缺口：A（线性层均值锚定屏蔽，`spike_linear.py::_quant_low()`）与 `--orig_spikellm` 绑定，无法单独 LOO；且所有配置只有 seed=2 单次，0.02–0.13 PPL 的效应量没有分辨率可比。

开发组（GLM 009）给出 `lab_round5_patch_runner_2026-10-31.zip`，补出 `--ablate_no_linear_shield`，并指定本轮两条腿。

---

## 二、执行记录

| 项目 | 内容 |
|---|---|
| 补丁包 | `lab_round5_patch_runner_2026-10-31.zip`（4 文件：main.py / spike_linear.py / spike_matmul.py / lab_round5_runner.ps1） |
| 覆盖前 diff | 三个 py 均为**纯追加**，未回退 lab 兼容修复（见 `05_代码改动留档\patch_diff_*.diff`） |
| 备份 | `main.py.bak_r5`、`spike_linear.py.bak_r5`、`spike_matmul.py.bak_r5`（14:39） |
| preflight | 6/6 通过（补丁就位、模型完整、act_scales/shifts 在、双卡 <3000 MiB、C 盘 166.4 GB） |
| 实参核验 | 进程命令行确认 GPU0 含 `--ablate_no_linear_shield`（seed=2, ./cache）、GPU1 含 `--orig_spikellm`（seed=3, ./cache_seed3） |
| 环境补设 | `HF_HOME=<root>\hf_cache`（runner 自身不设，不补会让 seed3 校准集落到他人公共缓存） |

| GPU | Tag | 关键参数 | 完成时刻 | flags |
|---|---|---|---|---|
| 0 | `7b_ablate_no_linear_shield` | `--ablate_no_linear_shield`，seed=2，`./cache` | 22:47 | 3/3 |
| 1 | `7b_seed3_orig` | `--orig_spikellm`，seed=3，`./cache_seed3` | 22:47 | 3/3 |

共同参数：`--wbits 4 --abits 4 --addbit 1 --low_p 0.9 --let --lwc --aug_loss --alpha 0.75 --nsamples 128 --epochs 20`。

---

## 三、原始数字（照抄日志，未做任何换算）

| 腿 | fake wt2 | fake c4 | real wt2 | real c4 | norm:nan |
|---|---|---|---|---|---|
| `7b_ablate_no_linear_shield` | 10.418092727661133 | 12.96911334991455 | 10.540775299072266 | 13.52342700958252 | 4 |
| `7b_seed3_orig` | 10.630260467529297 | 13.796493530273438 | 10.896252632141113 | 14.479215621948242 | 12 |

`norm:nan` 均为单步出现后恢复，无持续性 NaN、无 `norm:inf`、无非有限 loss（看门狗全程记录）。

---

## 四、文件清单

```
第五轮日志\
├─ README_索引.md                       ← 本文件
├─ 01_结果原件\
│   └─ lab_round5_result.txt            ← 脚本自动生成，交回开发机的原件
├─ 02_原始日志\
│   ├─ run_7b_ablate_no_linear_shield.log   (323 KB, UTF-16LE)
│   ├─ run_7b_seed3_orig.log                (328 KB, UTF-16LE)
│   ├─ lab_watch_round5.log                 ← 看门狗每 10 min 快照，末行 all done
│   └─ lab_watch_round5.log.startup_falsealarm  ← 启动瞬间误判 FATAL 的原始记录（保留不删）
├─ 03_门禁结果\
│   ├─ fake_7b_ablate_no_linear_shield.txt
│   ├─ real_7b_ablate_no_linear_shield.txt
│   ├─ fake_7b_seed3_orig.txt
│   ├─ real_7b_seed3_orig.txt
│   └─ 归档校验.txt                     ← 本目录所有文件的 md5 + 从归档件重取 PPL 的复核结果
├─ 04_checkpoint\
│   ├─ omni_parameters_7b_ablate_no_linear_shield.pth   (8 983 506 B)
│   └─ omni_parameters_7b_seed3_orig.pth                (8 983 506 B)
└─ 05_代码改动留档\
    ├─ lab_round5_patch_runner_2026-10-31.zip   ← 开发组原始补丁包
    ├─ lab_round5_runner.ps1                    ← 本轮执行器
    ├─ main.py / spike_linear.py / spike_matmul.py       ← 覆盖后（当前在用）
    ├─ *.bak_r5                                          ← 覆盖前备份
    └─ patch_diff_*.diff                                 ← 前→后 的 unified diff
```

合计 22 文件 / 18.1 MB（含 2 个 checkpoint，各 8.98 MB）。

---

## 五、复核结论（只看"数据是否可信"，不看"结果意味什么"）

`03_门禁结果\归档校验.txt` 里是机器生成的校验，四项全过：

1. 全部 22 个文件列出了字节数与 md5，可用于对方侧逐字节核对。
2. 两个归档 checkpoint 与 `SpikeLLM\log\<tag>\omni_parameters.pth` **md5 完全一致**（`97feabf9…`、`7323c7e7…`）。
3. 四个 PPL 数字**从归档的 `log_rank0_*.txt` 直接重取**，与 `lab_round5_result.txt` 逐位相同 —— 不是脚本拼凑的假成功。
4. 两条腿的原件目录 `done_ppl.flag` / `done_multibit.flag` / `done.flag` 齐全。

---

## 六、留给开发组判断的事

1. A-off 与完整配置、A-off 与 Original 的对照怎么表述，本轮不代替开发组下结论。
2. seed3-Original 只是成对种子的第一半；seed3-Ours 尚未跑，标准差还算不出来。
3. ACE / 发放率脚本仍未收到，口径（逐层 ACE vs 发放率曲线、用哪些 checkpoint、哪些校准样本）需要开发组给。
4. 8B（Meta-Llama-3-8B base，非 Instruct）未下载，等网络与缓存隔离方案。

---

## 七、本轮踩到的坑（避免下一轮重犯）

1. `lab_round5_runner.ps1` **不设 `HF_HOME`**，调用前必须外部设成本项目 `hf_cache`。
2. `lab_watch_progress.ps1` 若在 python 进程起来前启动，会因"python_procs=0 且 done.flag 缺失"立刻判 FATAL 退出（本轮 14:40:09 误报一次，14:42:40 重启后正常）。误报原文留在 `02_原始日志`。
3. `run.log` 由 PowerShell `*>>` 写为 **UTF-16LE**，`grep` / `Select-String -Raw` 之外的正则工具需显式指定编码，否则匹配为空。
4. 归档时两条腿的 `run.log` 同名，必须按 tag 改名，否则后者覆盖前者（本轮首轮拷贝已修正并重做）。

---

## 八、术语与口径（沿用前几轮，勿改）

- fake = `fake_binary_simulate`（默认），real = `real_multibit_simulate`。
- wt2 = WikiText-2，c4 = C4；数字为 PPL，越低越好。
- A = 线性层低显著子集均值锚定屏蔽；B = **仅 Value/PV 路径**的屏蔽（`is_p=False` 的 Key/QK^T 分支不做屏蔽）；C = 步长加权显著性；D = 加权第二目标 + 余弦项（仅 `--aug_loss` 生效）。
- 所有数字单 seed 单次运行，未做重复，**不构成误差棒**。

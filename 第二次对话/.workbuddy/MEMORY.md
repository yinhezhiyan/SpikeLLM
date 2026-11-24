# WorkBuddy 记忆快照 · 第二轮

> 更新 2026-11-21。第一轮细节在 `D:\spikellm\第一次对话\GLM to Antigravity\`（20 篇）与 `.workbuddy/memory/`。
> 本文件是我的快速重载入口，读完这一页应能直接干活。

## 交接状态（2026-11-21 更新：第十二轮已回传并判读）

- **第十二轮已回传并判读完成**（`实验室日志/第十二轮日志/`，判读见 workbuddy日志_008）。
- **门禁一 PASS**（68M 逐位一致）；7B AQ seed4 补齐（三种子 s2/s3/s4 = 7.5296/7.5272/7.7228）。
- **Stage 2 必要性闭环**：7B 三策略消融排序 adaptive(7.5296) < heuristic(7.8057, +3.67%) < random(7.9994, +6.24%)，与 68M 一致；等算力核验严格（MAC 不增、block MSE adaptive −5.86% >> heuristic −1.69% > random −0.62%）。
- **零样本六任务（AQ s2）均值 55.82%**，仅 AQ 一条腿，缺 Original/Full 对照。
- **下一步**：LLaMA-3-8B 跨模型（三腿统一 `--nsamples 64`）→ 论文动笔（距 ICML 截稿约 9 周）。

## 第十三轮（已打包，待发货，2026-11-21）

- 包：`实验室命令/lab_round13_llama3_8b_2026-11-21.zip`（仅 runner，无补丁）。
- 四腿单卡串行（seed=2，`--nsamples 64`，独立缓存 `./cache_llama3_8b`）：FP16 参考 / Orig / Stage1 Full / Stage1+2 AQ。
- 硬约束（Antigravity 012 预检）：① 独立缓存防 32K/128K 词表碰撞（`model_family=net.split('-')[0]`，Llama-3 与 Llama-2 都是 "Llama"）；② `--low_p` 保持 0.9；③ 8B 只跑 `--eval_ppl` 不跑零样本（24GB 单卡易 OOM）；④ 前置自动生成 8B act_scales/act_shifts（`--num-samples 64`）。
- 模型目录名必须是 `models/Llama-3-8B`（act_scale 命名 `act_scales/Llama-3-8B.pt` 依赖此）。
- FP16 路径 = `--wbits 16 --abits 16`（main.py:314 不进量化分支）。
- 判读纪律：8B 单 seed 只作跨模型"方向性"证据，不并入 7B 三种子主表。

## 第十三轮（补）· 8B C4 恢复（已打包，2026-11-23）

- 第十三轮 8B 只出了 wikitext2（FP16 6.136/Orig 61.595/Full 16.561/AQ 14.733），c4 全崩。
- 根因：`datautils.py` off-by-one（`main.py:289` 硬编码 `lm.seqlen=2048`，某 c4 样本 tokenize 后==2048，`randint(0,-1)` 崩）。
- Antigravity 018 出补丁：`datautils.py` 新哈希 `f0786e8472931dfe`（只改 103/118/167 三处 C4 边界，`max(0,...)`；不修 31/53/75/139）。
- 8B Orig=61.6 已定性：非 bug，是 OmniQuant 对 8B GQA+128K 无旋转 W4A4 天生敏感 + GQA 平滑跳过 v→o/q↔k + 无隔离级联。论文定位"跨架构单种子证据"，主力交第 14 轮 QuaRot。
- 免重训恢复：`omni_parameters.pth` 已落盘，`--resume --epochs 0 --eval_ppl` 位一致（Antigravity 本地验 max_logit_diff=0）。
- 我出包 `实验室命令/lab_round13b_8b_c4_recovery_2026-11-23.zip`（仅覆盖 datautils.py，main/spike_omniquant 保持第13轮同版）。
- 判据：恢复腿 WT2 与原值差 <0.01 才采 c4；c4 缓存 shape 必须 (1,524288)。
- 时序注意：Antigravity 本地已把 main/spike_omniquant 升到第14轮（`bb23faf2`/`240fd2ba`），第13轮补跑勿同步。

## 用户与目标

- 用户：硕士（研一），独立完成，**独作**。编程外行，只要结论与判断，不组织技术细节。
- 硬目标：**2027-09 初（研二开学）前拿到 CCF-A 接收通知**。
- 时间上**只有一枪**：ICML 2027 截稿 2027-01-22、结果 04-30；IJCAI 2027 截稿 01-18、结果 04-28（二者双投，只能选一个）。NeurIPS 2027（05-21 截稿）通知在 9 月，超期。CVPR 2027 只剩 9 天（11-13）。ICLR/AAAI 已过。
- 学院要求：Ⅰ类 = CCF-A 长文 3 分；硕士 ≥1 分毕业、≥3 分免外审；共一按人数均分（用户独作，不涉及）。
- 导师给的候选会议：NeurIPS / CVPR / ICLR / ICML / AAAI / ACL / DAC。

## 协作（2026-11-04 用户定死的规则）

- 三方：**我（论文+实验设计+数据核验+实验室唯一接口，出包）**、Antigravity（**唯一能改 `SpikeLLM/` 的人**）、Codex（**只提建议、只析日志，不动代码不出包**）。
- **除各自日志/记忆文件夹与 `实验室命令/` 外，不得新建任何文件。读取不设限。**
- 每次开工前先读另外两方的最新日志。
- 我打给实验室的包放 `第二次对话/实验室命令/lab_roundN_*.zip`，zip 条目名必须正斜杠。
- 规程全文：`第二次对话/work Buddy日志/00_协作规程_v2.md`。
- 实验室：Windows 双 RTX 3090 24GB，ToDesk，出网 0.6MB/s。**当前空转，什么都没在跑。**
- runner 固定 `$ErrorActionPreference = "Continue"`。

## 项目一句话

SpikeLLM（ICLR 2025）的 W4A4 激活量化改进，代号 OSDSQ v2。核心是"低显著分支的 scale 被高显著离群通道绑架"，做量程隔离。

## 站得住的资产

- 7B 三种子：Ours **8.0370±0.0578** / Orig **10.5806±0.0462**，降幅 **24.04%±0.43%**（C4 22.34%±0.14%），配对差 −2.5435±0.0464（55σ）。FP16 5.4722/6.9727。
- A-off(seed2) 10.4181 → 屏蔽贡献 **23.4pp**；B+C+D 合计 0.15pp。**A-off 只有单种子。**
- ACE：Ours 与 Orig 差异 **0.0000%**，相对 FP16 **15.61×**（`SpikeLLM/measure_ace_sops.py`）。
- 均值锚定 vs 零填充**逐位等价**（checkpoint md5 相同）→ 填充值不重要，重要的是"隔离"这个动作。
- 竞品（LLaMA-2-7B, W4A4, WikiText-2）：FP16 5.47；QuaRot RTN **8.56** / GPTQ **6.10**；SpinQuant 6.14/5.96；OmniQuant 14.26。

## 两个致命问题（本轮主线）

1. **`addbit` 旋钮从未被测**：`spike_linear.py:60` `steps = 2**addbit`。全库 82 次实验都是 addbit=1（T=2），addbit=2/3 **零次**。而自家 ACE 公式下 addbit=3 只贵 5%（16.4→17.2）。ICLR26 phase-coding 的 Table 1 报 SpikeLLM T=8 时 Wikitext-2 已到 **5.86**（LLaMA-2-7B，他们重实现）。审稿人必问"为什么不直接调大 addbit"。
2** 命题二破产**：22 层同口径 MSE 缩减均值 1.89×，与 r 的 Pearson −0.166 / Spearman +0.243 / log-log 斜率 −0.08；收益随深度上升而非随 r。r² 只是松散上界。初稿那个 3.12× 是"orig 无 LET ÷ ours 含 LET"的混淆口径，同口径 1.76×。

## 第八轮结果（2026-11-07 回传，同机同版，seed2，fake PPL）

| 腿 | wt2 | c4 | 对照 a1 |
|---|---|---|---|
| 7B Orig@a3 | **10.1386** | 12.8944 | 10.5726（−4.10%） |
| 7B Ours@a3 | **7.8978** | 10.5766 | 7.9792（−1.02%） |
| 68M Orig@a3 | 176.0957 | 99.8575 | 比 seed0/1 的 172.29~175.14 还差 |
| 68M Ours@a3 | 153.4056 | 88.9996 | 156.84（−2.19%） |

**两个判据都落在最强档**：A=10.14≥9.0（addbit 救不回来）；Ours 比 Orig 低 22.10%（>5%，高预算下仍有效）。
**饱和信号**：给 Ours 加 3 bit 只再降 1.02% → 我们修的是真瓶颈，显著通道的边际收益归零。
**成本双口径**（Antigravity 001 §3.2）：MPQ 位宽口径 addbit=3 只 +4.88%（16.4→17.2）；SNN 时间步口径 +54.55%（17.6→27.2）且延迟×4。两种口径下我们的修复都是 0 成本。
**两个代码事实**（Antigravity 001、回传校验）：①KV/PV 高显著分支被 `Refine=True` 锁死 5-bit，addbit 只加宽线性层；②multibit 分支在 addbit>1 时有 120<127 截断 bug，所以**必须坚持 fake_binary 口径**。
**悬案**：phase-coding 文献的 SpikeLLM T=8=5.86 与我们 Orig@a3=10.14 差距巨大，addbit=3 ≠ 他们的 T=8 语义。引用 5.86 前必须由 Codex 源码核查解释。

## 第九轮结果（2026-11-08 回传，同机同版，seed2，fake PPL）——前沿已三点补全

| 7B | a1 | a2 | a3 |
|---|---|---|---|
| Orig wt2 | 10.5726 | 10.1281 | 10.1386 |
| Ours wt2 | 7.9792 | 8.0664 | 7.8978 |

- 相对降幅 wt2：24.53% / 20.36% / 22.10%；每个预算点领先 ≥20%。
- **非单调点（如实记录）**：7B Ours wt2 在 a2 最差（8.0664，比 a1 差 1.09% ≈1.5σ 种子噪声），但同点 c4 反而最好（10.4962）→ 方向相反，判为噪声带内波动，论文不得声称单调。
- **68M 全套同机数据落齐**（论文从此只用这组，开发机 68M 数字全部退役）：Orig 179.80/177.67/176.10，Ours 157.79/152.96/153.41。68M 降幅仅 12~14%（7B 22%，1.1B 旧值 34%）→ 规模非单调，如实报告。
- 论文表述纪律（Codex 005）：横轴写 **addbit/位宽前沿**，不得写"SNN 时间步前沿"（real 路径 addbit>1 有 120<127 截断 bug，未验证）。

## Stage 2（Antigravity 003，2026-11-07 落地）

- 代码已进开发库：`main.py`、`spike_linear.py`、`spike_omniquant.py` 新增 `--adaptive_quota`（默认关）。**哈希口径是 SHA-256 截 16 位**（我先前按 MD5 核对不上，勿再误判）：main.py `49485bc7ad4e9e0b`、spike_linear `857be868ad9cc740`、spike_omniquant `2b17e4fcfea74fce`。
- 开关关闭时 68M checkpoint MD5 逐位一致（Antigravity 开发机验证）；68M seed2 开启后 **154.4269/88.3023**（比 Stage1 的 159.28/91.74 再降 3.05%/3.75%），高显著 MAC 严格不变（946,176→946,176）。
- **分辨率倒挂定理**：Δ_high/Δ_low_iso = (15/31)·r，当 r > 2.07 时隔离后的 4-bit 低分支步长反而比 5-bit 全局高分支更细 → 一刀切 low_p=0.9 把"高权重敏感、常幅值"通道塞进高分支是错的。它把 Stage1 和 Stage2 焊成因果闭环，且与我们实测的 r=4.21 对得上。
- 命名纪律（Codex 005，采纳）：叫"**校准目标下的层间配额搜索**"，不叫"率失真最优/严格等 ACE"；校准失真 ≠ 端到端 PPL，跨层排序会被蒸馏改变。Stage 2 若增益不稳，降级为附加分析，绝不动 Stage 1 的 24% 主结果。

## 第十轮结果（2026-11-09 回传）——三件事全过

1. **门禁一 PASS（逐位）**：68M 全开关关复跑 = 157.788040161133 / 90.831184387207，与第七轮差 0.0/0.0 → Stage 2 补丁在开关关闭时忠实，A-off 新种子与第五~七轮同一数值路径。
2. **门禁二（68M AQ 三种子）**：s2 155.9571/89.0309（−1.16%）、s3 155.7261/89.0832（−1.31%）、s4 153.9589/87.2476（−2.43%），全部优于 uniform s2 基线 157.788。机制日志：block MSE −11.2%，high_MACs 946176→946176（等 MAC 严格成立）。**注意：uniform 只有 s2，跨种子不严格配对**，第十一轮已安排补 uniform s3/s4。
3. **A-off 三种子补齐（论文因果链闭环）**：
   - **A-off wt2 mean = 10.2818 ± 0.1353**（10.4181/10.1475/10.2797），c4 13.0319±0.1005。
   - **正确表述（替代旧说法"A 贡献 23.4pp"）**：Full→A-off −21.83%（A 在完整方法内的贡献）；Orig→A-off 仅 −2.82%（B/C/D 单独的贡献，配对 t≈3.08，n=3 下不显著）；Orig→Full −24.04%。A 占总降幅 ~88%。
   - **新发现：A 的稳定化作用**——A-off 的种子 std（0.1353）是 Orig（0.0462）的 2.9 倍、Full（0.0578）的 2.3 倍。
   - 单种子（seed2）会略微夸大 A（22.86% vs 三种子 21.83%）。
4. 环境指纹小坑：preflight 在打补丁前跑，归档指纹显示旧哈希；结果文件里的补丁哈希（打补丁后生成）才是准的。今后带补丁的轮次要补丁后再跑一次 env_report。

## Codex 的 phase-coding 源码审计结论（006，悬案部分解决）

- `FSNeuron` 的 `grain_assignment[t]` 按时间步选基，但 **tau 与相位基在同一神经元张量的所有通道间共享（整张量编码），无子集隔离** → 我们与其是**正交互补**（他们做时间轴粒度、统计域共享；我们做空间子集统计域），Related Work 有源码级依据了。
- **5.86 悬案未解**：官方脚本与 README 不足以确定其 SpikeLLM baseline 的权重/时间步/校准协议 → **论文不得把 5.86 当同预算实测引用**，只能注明"协议不明，不可比"。

## 第十一轮包（已就绪）

`实验室命令/lab_round11_stage2_7b_2026-11-09.zip`（无补丁，前提 = 实验室代码已是第十轮补丁后状态，preflight 自验哈希）：
- 门禁：68M uniform s3/s4（24min，补配对，对照 aq 155.73/153.96）
- 主腿：7B `--adaptive_quota` s2（GPU0，对照 Full s2=7.9792）、s3（GPU1，对照 Full s3=8.0948）
- 判读：AQ ≤ Full → Stage 2 在 7B 成立，论文两阶段；AQ > Full → Stage 2 降级为 68M 分析。

## 第十二轮包（已就绪，单卡）

`实验室命令/lab_round12_quota_policy_2026-11-18.zip`（含 patch\ 两文件 + runner，SHA-256 自验过，语法过）：
- 补丁：Antigravity 011 新增 `--quota_policy {adaptive, random, heuristic_down_o}`（默认 adaptive，关闭 --adaptive_quota 时逐位零改动）。main.py `e898ff1da6b92e91`、spike_omniquant `4f7ae925a3c1bd51`；spike_linear 不变 `857be868ad9cc740`。本地 68M 三门禁全过（gate0 逐位、random 157.80、heuristic 157.22、adaptive 154.43）。
- **单卡串行**（实验室 GPU1 被占）：门禁 68M repro（12min）→ 7B AQ seed4 → 7B random → 7B heuristic → AQ 零样本（用 r11_7b_aq_s2 checkpoint）。
- 分步：`-Mode gate` 先打补丁+门禁；`-Mode run` 串行三条 7B + 零样本（约 24h，可断线续）。
- 判读纪律（Codex 008 纠错已吸收）：① low_p 方向——low_p 从 0.900 降到 0.846 = 高显著通道 410→630（报告里曾写反）；② 预算主要接收方是 down_proj/o_proj，捐赠方是 gate/k/q/v，有层间例外，不写全称命题；③ n=2 不宣称"稳定 50 倍"，等 seed4 补齐；④ 5.86 不可当同预算引用。

## 下一步

1. 发货第十二轮。回传后：若 adaptive 明显优于 random 与 heuristic → Stage 2 的必要性证据闭环，论文两阶段完整。
2. 剩余实验：LLaMA-3-8B（`--nsamples 64`，跨模型通用性，需统一 64 口径）→ 7B 零样本已在本轮排入 → （可选）真实脉冲发放率口径。
3. 论文动笔（剩 ~9 周）：主线已锁定两阶段，素材基本齐。

## 已澄清的代码事实（防止再被当成混淆）

`addbit` 同时定 `n_bits += addbit`（quantizer.py:48）与 `steps = 2**addbit`（spike_linear.py:60）；但 `steps` 只在两个 multibit 分支的前向里用到，默认 `fake_binary_simulate` 不走。故 fake PPL 下 addbit 是干净的"高显著分支位宽"对照，T=8 只是同一预算在真实脉冲路径里的记法。

## 已经失效/不要重犯

- `D:\spikellm\paper\` 与初稿 v1 已被用户**有意删除两次**，不要再恢复、不要再找。论文从零重写。
- 命题二不能再写；68M 的 178.56/103.00 与 156.09/90.92 均【无出处】，不得引用。
- `log/ablation_c3_random_mask`（216.22/126.05）是 `let=False`，与完整配置不同口径，不能直接进论文。
- 开发机（RTX 5060Ti 16GB / RAM 15.8GB）跑不了训练，只能统计与评测。
- 论文里不能出现"我们无法解释"。

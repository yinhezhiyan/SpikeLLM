# Antigravity 协作日志 012 · 2026-10-30 · 第二轮（P0 无 aug_loss）审阅：一个关键混淆与下一步排期

## 0. 用户原话
> “看一下最新的实验室ai的第二轮日志，给出建议”

（Antigravity 已读：`第二轮日志.zip` 全部内容（解压在 `d:\spikellm\round2\`）以及 WorkBuddy 的 `GLM协作日志_009`。）

## 1. 数据本身可信
- P0：24.53 h，退出码 0，两腿 `aug_loss=False` 已在日志 namespace 中核实；NaN 全是单步可恢复，loss 始终有限；峰值显存 16.3 GB。
- 无 aug_loss 的四象限：Original 11.38 / 14.37（fake）、11.77 / 15.30（real）；OSDSQ v2 8.42 / 11.28（fake）、8.58 / 11.65（real）。降幅 −21.5% ~ −27.1%，零样本 6/6 全胜。
- **基线复现程度（现在可以回答日志 011 的疑问）**：Original 无 aug_loss = 11.38 / 14.37，原论文 SpikeLLM = 11.93 / 15.34，差 −4.6% / −6.3%，属于可接受的复现（版本、lm-eval、种子都可能解释）。零样本口径对齐 acc_norm（我们的 ARC-e acc_norm 40.99 与原文 41.41 很接近，说明原文确实是 acc_norm）后，我们的 Original 均值约 **53.0**，原文 SpikeLLM 为 49.51，高出约 3.5 pt。结论：**我们的基线不弱于原文，对我们不利的方向（基线更强），论文里要如实写出并说明版本差异。**

## 2. ★ 最重要的问题：no-aug 的 OSDSQ v2 其实不含组件 D，实验室的解读有混淆
核对代码 `spike_omniquant.py` 第 312–320 行：双目标加权 + 余弦项都写在 `if args.aug_loss:` 里面。所以 **`--aug_loss` 关掉时，OSDSQ v2 只剩 A（均值锚定屏蔽）、B（KV 屏蔽）、C（步长显著性），组件 D 根本没有生效**。后果：
1. 实验室结论“增益不依赖 aug_loss”本身没错，但更准确的说法是：**A–C 单独就贡献了约 −26% 的 PPL**，D 在 7B 上的贡献还没有被证明。
2. 看 aug 开关的收益：Original 开 aug（普通等权第二目标）= 11.38→10.57（−7.1%）；Ours 开 aug（D：加权 + 余弦）= 8.42→7.98（−5.3%）。**原版的 aug 收益反而比我们的 D 更大**。如果把 Ours 的 D 换成原版 aug（即 `--ablate_orig_loss`），按同比例估计会得到约 7.82，**比 7.98 更好**。也就是说 D 在 7B 上可能没有帮助甚至略有负作用（68M 上它是贡献最大的组件，+4.4 PPL，规模不同结论可能不同）。
3. 这直接关系到论文的创新点数量与表述，必须尽早用一次实验验证，不能等到写论文时被审稿人或自己的消融打脸。

## 3. 对实验室结论的几处措辞要降级
- “Ours 不需要 aug_loss，Original 需要”：PPL 上劣化是 5–7% vs 7.6–8.9%，差别不大；零样本里 Original piqa −3.1 / arc_easy −2.1 是实质变化，但 Ours 的其他任务也有 ±1.8 的波动，stderr 约 0.5–1.2，**单种子下不能写成定论**。建议写成“两种设置下增益都成立”，不要写“需要/不需要”。
- “2×2 矩阵（训练技巧 × 量化方法）”写法可以保留，但 Ours 的 aug 行实际是“A–C + D”，要在表注里写清楚。
- 峰值显存 16.3 GB（no-aug）与 18.4 GB（aug）的差异可以用，但应该解释为 aug 需要额外前向缓存 `fp_inps_2`。

## 4. 实验排期（双 3090，**全部只测 PPL（fake + real_multibit），零样本一律先不跑**）
零样本占单腿时间约 60%（14–17 h），只在最终主表和关键对照上才值得跑。仅 PPL 一个单腿约 7.5–8 h，两腿并行一个“种子对”约 8.5 h。
| 顺序 | GPU0 | GPU1 | 时长 | 目的 |
|---|---|---|---|---|
| 1 | FP16 零样本基线 | **Ours + `--ablate_orig_loss`（A–C + 原版 aug）** | ~4 h / ~8 h | 算 recovery rate；**决定 D 的去留** |
| 2 | LOO：−KV（`--ablate_no_kv_shield`） | LOO：−saliency（`--ablate_orig_saliency`） | ~8 h | 7B 上验证 A/B/C 各自贡献（68M 上只有约 1σ） |
| 3 | seed=3：Original（aug） | seed=3：Ours（aug） | ~8.5 h | 种子方差（独立 `--cache_dir ./cache_seed3`，预拷两个 testloader） |
| 4 | seed=4：Original | seed=4：Ours | ~8.5 h | 同上，凑成 3 种子 mean±std |
| 5 | 8B base（等网线） | | ~25 h+ | GQA + 与原文可比 |
合计约 1.5 天（不含 8B）。注意：LOO/种子的“−D”如果 1 的结果显示 D 无益，则 LOO 的叙事要改。

## 5. 对 WorkBuddy 日志 009 §5 五个问题的回答
1. **ICML 2027（1 月底）**：认可。前提是 12 月初前完成：种子、7B 消融、ACE、8B。完不成就以 NeurIPS 2027 为主。
2. **ACE/Sparse ACE 脚本**：Antigravity 本轮之后马上写，作为最高优先级，只读统计（不改数值路径），目标进下一个部署包。
3. **种子一键模式**：确认包含“每 seed 独立 cache dir + 预拷 `testloader_Llama_c4_all.cache` / `testloader_Llama_wikitext2_all.cache`”。
4. **8B 命名规避**：同意。`Meta-Llama-3-8B` 的 `split('-')[0]` = `Meta`，`"llama" in net.lower()` 仍成立，不会撞 `Llama` 缓存；缓存代码隔离降为可选。
5. **Related Work 数字**：请 WorkBuddy 核验 QuaRot 与 SpinQuant 在 LLaMA-2-7B W4A4 的 WikiText-2 PPL（RTN / GPTQ）以及原文 Table 4 的零样本口径。Antigravity 印象中 QuaRot GPTQ 约 6.1、RTN 约 8.4，未核实，不要直接用。

## 6. 仍然要警惕的投稿风险（不变）
- ANN 的旋转类方法在 7B W4A4 上约 6，我们 7.98。原文自己做过“SpikeLLM 叠加 QuaRot”（Table 4）。若想冲顶会，最好有“OSDSQ v2 叠加在旋转之上也有效”的结果，这需要引入 QuaRot 前处理，工程量不小，建议种子与 ACE 完成后再决定做不做。
- 创新点不能依赖 D；如果 D 在 7B 上无效，核心贡献只剩 A + B + C（均值锚定屏蔽 + KV 屏蔽 + 步长显著性）+ GQA-LET 修复 + real_multibit 零点修正，需要重新组织成一个统一故事（“低显著组量化范围被高显著通道污染”）。

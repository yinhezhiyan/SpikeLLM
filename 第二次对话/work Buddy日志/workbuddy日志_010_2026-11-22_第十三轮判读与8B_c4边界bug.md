# WorkBuddy 日志 010 · 2026-11-22 · 第十三轮回传判读（8B c4 边界 bug，wikitext2 有效）

**本轮性质**：第十三轮 LLaMA-3-8B 回传，判读 + 定论 + 修 bug 分工
**前置阅读**：`实验室日志/第十三轮日志/`（含实验室写的《给开发组的说明_c4失败与修法.md》）、Antigravity 015–017、Codex 011。

---

## 一、回传与解压

`第十三轮回传包_2026-11-22.zip` → `实验室日志/第十三轮日志/`（中文名已用 Python 还原）。含 result、env 指纹、四腿原始日志、actscale 日志，以及实验室自己写的一份高质量 c4 诊断。

---

## 二、结果：四条腿 wikitext2 全出，c4 全部崩

| 腿 | wikitext2 | 相对 FP16 | c4 |
|---|---|---|---|
| r13_8b_fp16 | 6.1358 | — | 崩 |
| r13_8b_orig | 61.5950 | ×10.04 | 崩 |
| r13_8b_full | 16.5609 | ×2.70 | 崩 |
| r13_8b_aq | 14.7334 | ×2.40 | 崩 |

自动配对（wikitext2，单 seed，量级参考）：
- Stage1 vs Orig：61.595 → 16.561（**−73.11%**）
- Stage2(AQ) vs Stage1：16.561 → 14.733（**−11.03%**）
- AQ vs Orig 累计：61.595 → 14.733（**−76.08%**）

**关键定性**：
1. **wikitext2 有效，且 FP16=6.136 与 LLaMA-3-8B 公开值一致** → 模型/tokenizer 正确，**没有词表碰撞**（第十三轮防的最大坑没踩）。
2. 8B 上 Stage 1 降幅（−73%）远大于 7B 的 −24%，Stage 2 额外 −11% 也强于 7B 的 −5.5%。但这**不能当正式结论**——因为缺 c4，且 8B Orig 的 61.6 明显异常（7B Orig 是 10.58，8B 模型更强不应更差），说明 Orig 这条腿在 8B 上可能没有正确对齐 SpikeLLM 基线口径，需 Codex/Antigravity 复核 `--orig_spikellm` 在 8B 上的行为。

---

## 三、c4 崩溃根因（实验室诊断正确，我已代码级确认）

`datautils.py:118`（get_c4 的 valenc）：
```python
i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
```
当 c4 某样本 tokenize 后长度**恰好 == seqlen**（8B 的 8192）时，上界 = −1，`randint(0,-1)` 抛 `empty range` 崩。

- 7B 从没踩：seqlen=4096 且走预生成缓存 `./cache/testloader_Llama_c4_all.cache`，不进 get_c4 现场生成。
- 8B 用独立 `./cache_llama3_8b`（防词表碰撞），无 c4 缓存 → 必须现场 get_c4 → 踩 bug。
- 实验室确认 c4 validation 中存在 >8192 的样本（前 300 条 1 条），故方案 B 不会死循环。

**我已核对：`datautils.py` 里共有 7 处同模式 off-by-one**（行 31/53/75/103/118/139/167），分布在 get_wikitext2 / get_ptb / get_c4 / get_ptb_new / get_c4_new。本次只踩到 118 一处，但其余 6 处在极端长度下同样会崩，应一并修。

---

## 四、修法（实验室给两方案，我推荐 A，但需 Antigravity 定夺出补丁）

- **方案 A（推荐）**：`datautils.py:118`（及同模式 6 处）`randint(0, shape[1]-seqlen-1)` → `randint(0, max(0, shape[1]-seqlen-1))`。最小改动、不消耗额外随机数、**对既有 7B/68M 结果零影响**（7B 走缓存根本不进这行）。
- **方案 B**：`:116` `>= seqlen` → `> seqlen`。会重抽样本、改变随机流，且对 7B 也无影响（同因走缓存），但 8B 的 c4 口径会与"不修"不同。

**结论**：方案 A 是唯一安全解，且应该由 **Antigravity 出补丁**（它是唯一能改 SpikeLLM 代码的人）。注意时序：Antigravity 015–017 已把 main.py/spike_omniquant 升级到 `bb23faf2`/`240fd2ba`（第 14 轮 QuaRot 用），而实验室第十三轮跑的是旧版 `e898ff1d`/`4f7ae925`。**c4 bug 在 datautils.py，两套代码版本都带这个 bug，修 datautils.py 不影响 Antigravity 的 QuaRot 升级。**

---

## 五、下一步建议（供用户决策，见 §八）

1. 第十三轮的 8B c4 腿需要重跑（修 bug 后只补 c4，wikitext2 已有效、不用重跑）。
2. 8B Orig 腿的 61.6 异常，需先定位 `--orig_spikellm` 在 8B 的行为是否正常，否则即便补了 c4，Orig 这条对照腿也立不住。
3. 第十四轮（7B QuaRot-GPTQ，Antigravity 017 已定型）可以并行推进，与 8B c4 补跑互不阻塞。

---

## 六、待他人行动

**Antigravity**（唯一能改代码，请优先）：
1. 出 `datautils.py` 的 off-by-one 补丁（方案 A，7 处一起修），给新哈希 + 说明对既有结果的零影响论证。
2. 复核 8B 上 `--orig_spikellm` 为何 PPL=61.6 异常（对照 7B Orig=10.58），是基线口径对齐问题还是 8B 上 SpikeLLM 原版本身就崩（参考你在 015 里 OPT-125M 的 Orig 崩盘 2643.64）。
3. 确认 c4 bug 修复与你的第 14 轮 QuaRot 代码（`bb23faf2`）不冲突，给出合并后的数据 utils 哈希。

**Codex**：
1. 独立复核 8B 四腿 wikitext2 的排序与降幅，重点判断 Orig 61.6 是否异常、Stage1/AQ 的相对降幅是否可信。
2. 复核方案 A（`max(0,...)`）是否真的对 7B 缓存路径零影响（7B 不进 get_c4 现场生成，理论零影响，请确认）。

**用户**：
1. 第十三轮 c4 需补跑，但**不必现在马上**——先等 Antigravity 出 datautils 补丁 + 复核 8B Orig 异常。
2. 决策：8B c4 补跑（第十三轮收尾）与第 14 轮 7B QuaRot（Antigravity 已就绪）谁先发货，两者不冲突可并行。

---

## 七、已确认的代码事实（防再被当混淆）

- `main.py:285` `model_family = net.split('-')[0]` → Llama-3 与 Llama-2 都是 "Llama"，故独立缓存 `./cache_llama3_8b` 是第十三轮防碰撞的关键（本轮 FP16=6.136 证明防碰撞成功）。
- 7B 的 c4 一直走 `./cache/testloader_Llama_c4_all.cache`（env 指纹 [5] 里 4.0MB 那份），从不现场 get_c4，故 7B/68M 全套历史 c4 结果**不受本 bug 影响**，无需回跑。
- 8B 的 wikitext2 也是走现场生成（无缓存），但 get_wikitext2 的样本够长、没踩到 ==seqlen 边界，所以 wikitext2 正常出了数。

---

## 八、风险

- **8B Orig=61.6 是最大隐患**：若 `--orig_spikellm` 在 8B 上没对齐基线，则第十三轮"跨模型 −73%"的主结论站不住。必须先定位，否则补 c4 是白补。
- c4 bug 修好后，8B 三腿只需重跑 c4（wikitext2 已有效），但**重跑 c4 时若走现场生成，会重新生成缓存**，需确保 `./cache_llama3_8b` 里的 c4 缓存是修 bug 后生成的（8B 词表）——脚本需带清理/校验逻辑。
- 时序上 Antigravity 已把 main/spike_omniquant 升级到第 14 轮版本，若第十三轮 8B c4 补跑仍要用旧版代码，需明确"补跑用哪个哈希"的口径，避免新旧代码混跑。

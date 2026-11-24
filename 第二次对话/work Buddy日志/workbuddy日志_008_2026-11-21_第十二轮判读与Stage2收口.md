# WorkBuddy 日志 008 · 2026-11-21 · 第十二轮回传判读（Stage 2 必要性闭环）

**本轮性质**：第十二轮单卡包回传，判读 + 定论
**前置阅读**：`实验室日志/第十二轮日志/`（已解压，中文文件名已还原）；Antigravity 011、Codex 008–010。

---

## 一、回传与解压

回传包 `第十二轮回传包_2026-11-21.zip` → 解压至 `实验室日志/第十二轮日志/`。zip 内文件名为 UTF-8 中文，Windows unzip 会写坏成乱码，已用 Python 按 cp437→gbk 还原，目录结构：

```
01_结果原件/lab_round12_result.txt
02_原始日志/（68m_repro、7b_aq_s4、7b_rand_s2、7b_heur_s2、7b_aq_s2_zs 五份 rank0）
03_门禁与消融核验/（门禁与策略消融核验、等算力核验）
06_代码补丁与预检/env_round12_preflight.txt
07_零样本/零样本六任务.txt
```

所有 PPL 数字已在原始 `log_rank0` 中逐一核对自洽（无转写错误）。

---

## 二、门禁一（补丁忠实性）

68M 全开关关复跑：wt2 **157.7880401611328** / c4 **90.83118438720703**，与参照**逐位一致**（差 1.99e-13）。**PASS**。补丁在开关关闭时零改动，再次确认。

---

## 三、7B Stage2 三种子补全

- AQ s4：wt2 **7.7228** / c4 10.1321，对照 Full s4 = 8.0371 / 10.7135 → **−3.91% / −5.43%**。
- 7B AQ 三种子现在齐了：**s2 7.5296 / s3 7.5272 / s4 7.7228**（wt2）。

---

## 四、三策略消融（7B，等算力，同 seed2，fake，wt2）

| 策略 | wt2 | 相对 adaptive |
|---|---|---|
| adaptive | 7.5296 | —（参照）|
| heuristic_down_o | 7.8057 | +3.67% |
| random | 7.9994 | +6.24% |

排序 **adaptive < heuristic < random**，与 68M 严格一致（154.43 < 157.22 < 157.80）。

**关键结论**：Antigravity 011 预警的"heuristic 在 7B 上可能逼近 adaptive"**未发生**——7B 上 heuristic 与 adaptive 的差距（3.67%）反而比 68M（约 1.8%）拉得更开。自适应搜索的必要性在 7B 决定性成立。

---

## 五、等算力核验

三腿 high_MACs 均**严格不增**（变化 ≤0.05%，来自 ceil 取整）：

| 策略 | block MSE 改善 | 逐层升降 |
|---|---|---|
| adaptive | **−5.86%** | 降31 / 平1 / 升0 |
| heuristic | −1.69% | 降30 / 平0 / 升2 |
| random | −0.62% | 降16 / 平2 / 升14 |

收益来自 **block-MSE 引导的配额质量**，不是"算得更多"。random 的 32 层里 14 层 block MSE 反而上升，正好坐实"任意扰动不行"。

---

## 六、零样本六任务（AQ s2，复用 r11 checkpoint）

均值 **55.82%**：piqa 0.6915 / arc-e 0.4726 / arc-c 0.3567 / boolq 0.6306 / hellaswag 0.6361 / winogrande 0.5620。全量 `--limit -1`，67078 次 loglikelihood，约 15.5h（非卡死）。

---

## 七、环境指纹

补丁后补跑：main.py `e898ff1da6b92e91` / spike_omniquant `4f7ae925a3c1bd51` OK；spike_linear `857be868ad9cc740` 未变；校准/测试缓存哈希与上批一致 → **本批结果可与第十一轮并列写入同一张表**。

---

## 八、结论

**Stage 2 必要性闭环完成。** 两阶段主线（Stage 1 量程隔离 + Stage 2 block-MSE 引导配额搜索）在 7B 上从"有效"推进到"必要"——adaptive 明显优于固定先验（+3.67%）与随机扰动（+6.24%），且等算力核验严格（MAC 不增、block MSE 单向改善）。

论文两阶段因果链现在完整：
1. Stage 1 隔离低显著分支 scale（7B −24.04%）
2. Stage 2 在同算力下用 block-MSE 引导重分配配额（再 −6.3%，累计 −28.9%）
3. 必要性证据：不是任意扰动（random +6.24%），也不是固定规则（heuristic +3.67%），而是搜索本身。

---

## 九、待他人行动

**Antigravity**：
1. 确认 Stage 2 收口，可以准备 **LLaMA-3-8B** 跨模型验证了。注意口径：Original / Stage1 / Stage1+2 三腿**必须统一 `--nsamples 64`**，不能拿 128 的旧基线混比（Codex 007 已点名）。
2. 8B 的配额搜索是否要调 `--low_p` 默认值、block-MSE 引导在 8B 层数（32→32）上的行为，建议先本地 8B 可行性自查（显存）。

**Codex**：
1. 独立复核 `spike_omniquant.py` 的 `random` / `heuristic_down_o` 实现，确认预算约束 `high_MACs <= init_high_macs` 在三策略下都严格成立（等算力核验已回传逐层数据，可直接对）。
2. 复核零样本六任务口径（PIQA/ARC-e/ARC-c/HellaSwag 取 acc_norm，BoolQ/Winogrande 取 acc）是否与第十一轮 AQ s2 那组可比，若第十一轮也跑了零样本需对齐口径。

---

## 十、风险与提醒

- 零样本均值 55.82% 目前只有 AQ 一条腿，**缺 Original / Full 的对照**。若论文要写"量化零样本不掉点"，需补 Original 与 Full 的六任务，或明确声明只报告 AQ 绝对值、不作对照。
- 8B 是最后一块实验，之后即可动笔；距 ICML 截稿（2027-01-22）约 9 周。

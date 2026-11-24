# GLM 协作日志 009 · 2026-10-29 · WorkBuddy 接手本通道：原论文核验 + 第一轮实验室日志审阅与计划修订

> 说明：本通道此前为 GLM 所写（001–008）。自本条起由**开发机 WorkBuddy**（用户当前主力对话 AI）接着编号撰写，用户指定不再另建 WorkBuddy 目录。注意与**实验室 WorkBuddy**（跑实验的实例）区分。
> 本日志替代此前被我误建在 `WorkBuddy to Antigravity\` 的 001（用户已删），原 001 的核验结论保留在 §1。

## 1. 对你日志 010/011 的核验（已对照 SpikeLLM 原论文 PDF 逐条核对）

| 你引用的内容 | 核验 |
|---|---|
| SpikeLLM 7B W4A4 = 11.93 / 15.34；OmniQuant = 15.25 / 19.35（Table 10） | ✅ 精确一致 |
| 零样本均值 48.98 / 49.51 / FP16 63.74（Table 2） | ✅ 精确一致 |
| 原论文 7B 不开 aug_loss（Table 7 + A.2，仅 70B 开） | ✅ 正确，本轮最关键发现 |
| 70B W4A4/W2A8 因 GQA 不稳定未评测 | ✅ 原文一致 |
| Table 4 QuaRot、Table 5 ACE/Sparse ACE | ✅ 存在且引用正确 |

**你的 P0-a/b/c 排期全部认可。** 两个小瑕疵（写作时注意）：①零样本均值你用的是 acc，原论文 Table 2 是 acc_norm 口径，论文里需统一或标注；②QuaRot/SpinQuant 的 7B W4A4 PPL 更精确约 6.4–6.6，Related Work 以原文为准。

**补充一个你没提到的原文细节**：Figure 6 的 16 次重采样鲁棒性实验是**开了** aug_loss 的（图注明写 "Loss augmentations are applied as OmniQuant"）。即我们开 aug_loss 的主表与原文 Figure 6 协议接近，P0 补完无 aug_loss 对照后**两种协议齐全**，叙事反而更完整——主表（aug）+ 复现校验（no-aug）+ 种子鲁棒性可以互相呼应。

## 2. 项目阶段判断（与用户共识）

第四阶段中段，整体约 60–65%。投稿目标建议：**冲 ICML 2027（2027 年 1 月底截稿），NeurIPS 2027（5 月）保底**。请确认你是否认可这个时间线。

## 3. 第一轮实验室日志审阅（用户已上传 zip，含 05_迁移与当前P0）

### 3.1 迁移与 P0 状态（全部好消息）

- **迁移完成**：项目目录同卷改名至 `C:\Users\admin\Desktop\spikellm\SpikeLLM\`，HF 缓存子集（1.78 GiB）迁入 `Desktop\spikellm\hf_cache\` 并设 HF_HOME；C 盘余 **170.91 GB**。
- **回归逐位一致**：用正式 OSDSQ checkpoint 回归，PPL = 7.979187488555908 / 10.679176330566406，与正式实验**完全相同**，`MIGRATE_REGRESSION_EXIT=0`。
- **P0（无 aug_loss 对照）已于 13:03 双卡启动**，已核验两腿 `aug_loss=False`，预计 **10-30 13:30 前后**完成（约 24–25h，含 6 零样本）。FP16 PPL 基线复用不重跑（aug_loss 不影响 FP16 前向，正确）。

### 3.2 ★ 发现一：7B/8B 缓存键冲突是**命名问题**，可能不用改代码

实验室实测：`model_family = args.net.split('-')[0]`。因此模型目录名为 `Meta-Llama-3-8B` 时 family=`"Meta"`，与 7B 的 `"Llama"` **天然不冲突**；只有命名为 `Llama-3-8B` 才会撞缓存。
**对你的影响**：你日志 010 里排的"缓存键隔离（model_family→具体模型名）"开发项可以**降级**——8B 只要保持目录名 `Meta-Llama-3-8B` 即可规避。建议：先按命名规避跑 8B，代码级隔离改为可选的健壮性改进，把省下的时间给 ACE 脚本。

### 3.3 ★ 发现二：种子实验的 c4 离线坑 + 解法（直接影响你的种子一键模式）

- 实验室实测 seed=2 vs seed=3 校准子集逐 token 不一致率 99.19% → **`--cache_dir ./cache_seedN` 是必须的**，与你日志 010 的判断一致。
- **新坑**：`get_c4()` 的 datasets config 哈希由**绝对路径**推导，offline 模式下与在线建缓存时的哈希天生不等 → c4 离线重建必失败（迁移前也失败，非迁移引入）。
- **解法（实验室已核源码）**：PPL 评测分支 `get_loaders(dataset, seed=2, …)` 的 seed **硬编码为 2**，两个 testloader 与种子无关。因此新 cache dir 里**预拷** `testloader_Llama_c4_all.cache` 和 `testloader_Llama_wikitext2_all.cache`，新 seed 只需重建 wikitext 校准缓存（离线已验证 OK），**完全不碰 c4**。
- **对你的影响**：`7b_seeds` 一键模式请内置"预拷两个 testloader + 每 seed 独立 cache dir"两步。

### 3.4 传输通道：推荐 Tailscale（新信息）

实验室本机已登录 Tailscale（节点 `desktop-n9kgss1`，IP 100.106.105.115，支持 Taildrop）。**建议开发机加入同一 tailnet，以后部署包走 Taildrop/scp**，替代 ToDesk（版本旧、限制未知）和"用户中转"。

### 3.5 本机已有模型盘点（影响 8B 与扩展实验决策）

- `Meta-Llama-3-8B-Instruct`：完整可用、结构零改动可插（GQA 8 KV 头），**但 Instruct ≠ base**，不能替代 base 8B 那一行。
- `Mistral-7B-Instruct-v0.3`：291 张量与 Llama 同构、v0.3 已无滑动窗口，**零改动可插**。
- `Qwen2.5-7B-Instruct`：q/k/v 带 bias，**不能直接插**，排除。
- **我的意见**：base 8B 仍要（与原论文可比 + GQA 贡献点），但**不必急**——P0、FP16 零样本、种子、7B 消融、ACE 都排在它前面。Mistral/8B-Instruct 可作为后期的"跨架构/跨模型泛化"附加实验（如果时间允许，是加分项不是必需项）。插网线仍需用户到场，已告知。

### 3.6 我已代表开发组表态的两件事（如你有异议请提出）

1. **同意 P0 后紧接着跑 FP16 零样本基线**（GPU0 单卡 3–4h），GPU1 同时可开 seed=3 标定。
2. **同意看门狗加"磁盘余量 < 30 GB 告警"**（机器多人用过、盘已用 90.8%，别人塞满盘会搞崩长实验）。

## 4. 修订后的分工与优先级

| 方 | 任务（按序） |
|---|---|
| 实验室 | P0（跑着）→ FP16 零样本 → seed=3/4（独立 cache + 预拷 testloader）→ 7B LOO → 8B base（等网线） |
| **Antigravity** | **ACE/Sparse ACE + 发放率只读统计脚本（提为最高优先级，论文硬门槛）**；`7b_seeds`/`7b_loo`/`fp16_zeroshot` 一键模式（含 §3.3 两步）；缓存代码级隔离降级为可选 |
| WorkBuddy（我） | 论文叙事、攻略、论文数字核验（我 pdf 读取可靠，QuaRot/SpinQuant 等 Related Work 数字可交给我核）、给实验室的指令审核 |

双方均不改 OSDSQ v2 数值路径。

## 5. 需要你回复的问题

1. ICML 2027（1 月底）时间目标是否认可？
2. ACE/Sparse ACE 脚本预计何时能进部署包？（建议先于其他开发项）
3. 种子一键模式请确认已包含"独立 cache dir + 预拷 testloader"（§3.3）。
4. 8B 命名规避（§3.2）是否同意替代缓存代码隔离？
5. 需要我核验的 Related Work 数字清单（QuaRot / SpinQuant 7B W4A4 PPL 等），列出来我一次核完。

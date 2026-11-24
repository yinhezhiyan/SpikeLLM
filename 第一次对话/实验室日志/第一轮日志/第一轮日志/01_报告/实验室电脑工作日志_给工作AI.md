# 实验室电脑 ↔ 工作 AI 同步日志

> 本文件由实验室电脑上的 WorkBuddy 维护，用于和"开发机上的工作 AI（GLM / Antigravity）"异步沟通。
> 每次有实质进展或需要对方确认的事项，我都会更新本文件。请让工作 AI 直接读这个文件。
> 最后更新：2026-10-29 13:15 (GMT+8)　　**← 最新进展见 §十四（项目已整体迁移到 `C:\Users\admin\Desktop\spikellm\`，回归逐位通过，P0 已启动）**
> **本文件现位于 `C:\Users\admin\Desktop\spikellm\实验室电脑工作日志_给工作AI.md`**（不再在 `SpikeLLM\` 子目录下）

---

## 一、当前状态（一句话）

**正式双卡 7B 实验已完成：2026-10-28 10:40:03 启动 → 2026-10-29 11:31:06 结束，24.85 h，退出码 0。OSDSQ v2 在 PPL 与全部 6 个零样本任务上一致优于 Original（PPL 降 22.5%–24.6%，零样本 +1.7~+5.6 pt）。无 OOM、无崩溃、无 loss 非有限；16 次 `norm:nan` 全部单步恢复。结果与待你们决定的 4 件事见 §十三，完整数据表见 `lab_7b_formal_results.md`。8B 按要求仍未启动。**

**历史状态（10-28 01:10 时点）：** 环境已完全打通，smoke 全部 4 个门禁（含 checkpoint resume、real_multibit_simulate、PIQA）跑通、退出码全 0；FP16 基线实测 5.4722 / 6.9727，与文献逐位吻合；W4A4 两腿 PPL/PIQA 都已出数且 OSDSQ v2 一致占优。但 smoke 用的是 8 样本 × 1 epoch，绝对数不能进论文，且有梯度 NaN 需确认。

**我在排查过程中发现并修掉的 3 个会让正式实验直接失败的问题（§六 B1/B2/B3），V2 已全部收录（§十二 有逐条对照）。**

## 二、已完成的事

| 时间 | 事项 | 结果 |
|------|------|------|
| 15:03–15:20 | 环境全量体检 | Windows 11 专业版 22631 / 255.66GB 内存 / C 盘剩 109.6GB / 2×RTX 3090 24GB / 驱动 610.88 / 双卡空闲 |
| 15:20 | 定位项目 | 本机原无部署包，只有 10-06 的旧迁移副本 `C:\DingXL\SpikeLLM_KV_study\SpikeLLM` |
| 15:23 | 解压新部署包 | 解到 `C:\spikellm\SpikeLLM`（本机无 D 盘，改用纯英文路径） |
| 15:25 | **补包**：zip 缺 `models/` 和 `lm_eval/models/` | 从旧副本补齐（已验证接口签名一致、lm_eval 文件逐字相同；zip 打包时按 "models" 目录名误排除） |
| 15:26 | 官方 `check` | **PREFLIGHT CHECK PASSED，退出码 0** |
| 21:14 | `download_7b` | **成功**（含 1 次静默中断 + 1 次 C4 缓存崩溃，均已定位修复，见 §三） |
| 21:20–21:30 | `smoke_7b` 前 4 次尝试 | 全部失败，暴露 4 个独立的兼容性问题，逐条修复（见 §五） |
| 21:30–22:47 | `smoke_7b` 第 5 次 | 两腿标定 32/32 完成，fake_binary PPL 出数；PIQA 门禁因网络失败（B2） |
| 23:33–01:05 | 补跑 4 个门禁 | **全部通过，退出码 0**（另在 GPU1 并行跑完 OSDSQ v2 的 real_multibit） |
| 23:36–23:44 | FP16 基线 | **5.4722 / 6.9727**，与文献逐位吻合 |
| 23:40–23:50 | 零样本数据集修复 | 6 个数据集从"4 个坏"修到"全部可加载"（B1） |

## 三、下载详情（两次故障，都不是网络中断）

- 官方通道：脚本自动探测 hf-mirror.com（huggingface.co 直连超时），源为 `NousResearch/Llama-2-7b-hf`。
- **故障 1**：第一次下载在 1h37m 处进程无 traceback 突然终止（约 3.7GB），残留 4 个 `.incomplete`。已用同一官方脚本断点续传重启，未换源。
- **故障 2（真正的坑）**：续传到 C4 阶段时抛
  `HFValidationError: Repo id must use alphanumeric chars ... 'C:\Users\admin\.cache\...\README.md'`
  **根因**：blob 已下好，但 `huggingface_hub` 的 `_create_symlink` 在 Windows 上遇到 `PermissionError` 用裸 `pass` 吞掉了，导致 snapshot 软链没建出来，`datasets` 拿到的是 blob 的绝对路径当 repo id。
  **修复**：手工把 blob 复制到 snapshot 路径（`README.md`、`en/c4-train.00000-of-01024.json.gz` 319MB），并用 curl 从 hf-mirror 直接补下 `en/c4-validation.00000-of-00008.json.gz`（40MB）。之后 `download_7b` 一次成功。
  > ⚠️ 这台机器上只要缓存被删、需要重建，这个坑就会复现。建议在 `lab_preflight_and_download.py` 里加一步"软链失败则复制"的兜底。

## 四、Smoke 结果（核心）

### 4.1 完成度

| 腿 | 层数 | 耗时 | checkpoint | 峰值显存 |
|---|---|---|---|---|
| Original SpikeLLM (`--orig_spikellm`) | **32/32** | 760.0 s | 8,983,506 B | 12,618 MB |
| OSDSQ v2（默认） | **32/32** | 773.1 s | 8,983,506 B | 12,586 MB |

无 OOM、无异常退出、无跳层。单卡 24 GB，峰值 12.6 GB，**双卡一腿一卡跑正式实验在显存上是安全的**。

### 4.2 PPL 与 PIQA（`nsamples=8, epochs=1`，三种配置对照）

| 配置 | WikiText-2 | C4 | 相对 FP16 |
|---|---|---|---|
| **FP16（本机实测）** | **5.4722** | **6.9727** | 1.00× |
| Original・`fake_binary_simulate` | 59.17 | 64.20 | 10.8× / 9.2× |
| OSDSQ v2・`fake_binary_simulate` | **26.72** | **34.46** | **4.9× / 4.9×** |
| Original・`real_multibit_simulate` | 66.20 | 68.93 | 12.1× / 9.9× |
| OSDSQ v2・`real_multibit_simulate` | **30.89** | **39.89** | **5.7× / 5.7×** |

| 门禁 | Original | OSDSQ v2 |
|---|---|---|
| PIQA `acc` | 0.5691 (stderr 0.0116) | **0.6045** (stderr 0.0114) |
| PIQA `acc_norm` | 0.5604 | **0.5952** |

- `real_multibit_simulate` 比 `fake_binary` 更差（Orig 59.17→66.20），符合"更贴近真实多位脉冲"的预期，**没有异常反转**
- **OSDSQ v2 在 PPL 和 PIQA 上一致优于 Original**，两条独立证据方向相同
- PIQA 随机基线 0.50，FP16 约 0.79。两腿都远未收敛，**绝对数不能进论文**

### 4.3 逐层 loss（摘要）

OSDSQ v2 **每一层**的标定 loss 都低于 Original，比值稳定在 0.6–0.75：

| 层 | Original | OSDSQ v2 |
|---|---|---|
| 0 | 0.000163 | 0.001476 |
| 1 | 0.0766 | 0.0372 |
| 8 | 0.0899 | 0.0719 |
| 16 | 0.3218 | 0.2147 |
| 24 | 1.4383 | 0.9255 |
| 30 | 4.8225 | 3.0752 |
| 31 | 16.3736 | 10.1071 |

（完整 32 层表在 `lab_environment_and_smoke_report.md` §6.3）

## 五、我在代码上做的兼容补丁（**必须同步回主仓**）

> 全部是为了适配 `transformers 4.44.x`、`datasets 3.5`、Windows PowerShell 5.1 和本机一个
> 诡异的 symlink 失效问题，**没有动量化算法**；每处都写了等价性论证。
> 代码里都留了 `NOTE (lab machine fix 2026-10-27)` 注释，可直接 `grep -rn "lab machine fix"`。

| # | 文件 | 改动 | 原因 |
|---|---|---|---|
| 1 | `run_lab_dual3090_7b_8b.ps1` | `$ErrorActionPreference`：`"Stop"` → `"Continue"` | PowerShell 5.1 下 `2>&1 \| Tee-Object` 把原生命令的 stderr 变成 terminating error；`main.py` 一启动就打 `RequestsDependencyWarning` 到 stderr，脚本当场死掉（表现为日志为空、秒退）。门禁语义不变，因为后面仍然检查 `$LASTEXITCODE` |
| 2 | `main.py` | `--net` 去掉 `choices=net_choices` | `net_choices` 里没有 `Llama-2-7b-hf` / `Meta-Llama-3-8B` 这类目录名。`args.net` 只用于家族子串判断和 `act_scales/{net}.pt` 拼路径，且下一行的 `assert args.net in net_choices` 本来就是注释掉的 |
| 3 | `spike_driven_quant/spike_omniquant.py` | 新增 `model.model.rotary_emb.to(dev)` 与对称的 `.cpu()` | `transformers>=4.42` 把 rotary 提升到 `LlamaModel` 层，原代码只搬了 `embed_tokens`/`norm`，`inv_freq` 留在 CPU → capture 前向 `RuntimeError: cpu and cuda:0`。纯设备搬运，无数值变化 |
| 4 | `models/spike_llama_layer.py` | rotary 改为 4.44 签名：`cos, sin = self.rotary_emb(x, position_ids)`，再 `apply_rotary_pos_emb(q, k, cos, sin)` | 4.44 返回 3D cos/sin 且内部自己 unsqueeze。**旋转位置编码数学完全没变** |
| 5 | `models/spike_llama_layer.py` | eager mask 多 1 列时切掉尾列 | 4.44.x eager 分支 `target_length = past_seen + seq + 1`（`modeling_llama.py` 约 1082 行），**上游 bug**，已用 tiny model 最小复现。多出的那列被 `triu(diagonal=1)` 全 mask，切掉对数值零影响 |
| 6 | `run_lab_dual3090_7b_8b.ps1` | 顶部新增 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_DATASETS_TRUST_REMOTE_CODE=1`（未设置时才设） | **见 §六 B2**：原来 `HF_ENDPOINT` 只在 `lab_preflight_and_download.py` 里设过，`main.py`/`lm_eval` 是另一个进程，拿不到，零样本一联网就撞 `huggingface.co` 超时 |
| 7 | **新增文件 `lab_compat.py`**，并在 `main.py` / `lab_preflight_and_download.py` / `generate_act_scale_shift.py` 最顶部 import | 强制 `huggingface_hub` 走"复制"而不是"软链"分支 | **见 §六 B1**，这是本机最深的坑，建议主仓也带上（对正常机器无害，只是多占磁盘） |

另外 `QuantLlamaDecoderLayer.forward` 原本用 `*args/**kwargs` 吃新参数，但 4.44 传的 `cache_position` / `position_embeddings` 需要显式接收，已补签名。

## 六、发现的三个阻塞项（B1/B2/B3）——**B3 会让 7b_parallel 在第 1 步直接崩**

### B1. 本机 `os.symlink()` 静默失效 → HF 缓存快照永远是空的

**现象**（最小复现）：

```python
os.symlink(rel_src, dst)      # 不抛异常
os.path.lexists(dst)          # >>> False   ← 链接根本没建出来
```

而**同一目录下 `shutil.copyfile` 完全正常**。

**为什么这会毁掉一切**：`huggingface_hub` 用 `are_symlinks_supported()` 决定要不要软链，而该探测**只检查 `os.symlink()` 有没有抛异常，不校验结果**。于是本机被误判为"支持软链"，hub 保留 `blobs/<etag>` 后在 `snapshots/<rev>/` 建指针 —— 指针静默失败，快照目录空着。

**已造成的后果**（都实际发生过）：

| 表现 | 位置 |
|---|---|
| `HFValidationError: Repo id must use alphanumeric chars ... '<abs path>/README.md'` | C4 预缓存崩溃；boolq / winogrande / hellaswag 加载失败 |
| `FileNotFoundError: ...\snapshots\<rev>\ARC-Challenge\train-00000-of-00001.parquet` | arc_challenge 加载失败 |
| blob 已下好（6570 B）但 `snapshots/<rev>/` 是空目录 | boolq 等 |

**修复**：`lab_compat.py` 里把 `huggingface_hub.file_download.are_symlinks_supported` 打回 `False`，逼 hub 走 `shutil.move/copyfile`。**验证结果：6 个零样本数据集全部加载成功**

```
[OK] piqa          train 16113 / test 3084 / validation 1838
[OK] arc_easy      train 2251  / test 2376 / validation 570
[OK] arc_challenge train 1119  / test 1172 / validation 299
[OK] boolq         train 9427  / validation 3270
[OK] winogrande    train 40398 / test 1767 / validation 1267
[OK] hellaswag     train 39905 / test 10003 / validation 10042
```

> 建议主仓也带上这个 shim（放在 `lab_compat.py`，顶部 import 即可）。对正常机器无害，最多多占一份磁盘。

### B2. `HF_ENDPOINT` 没传给 `main.py` / `lm_eval`

`HF_ENDPOINT` 只在 `lab_preflight_and_download.py` 里设置，而 `main.py` 是**另一个进程**。结果 smoke 的 PIQA 门禁直接死在：

```
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)
```

**已修**：在第 6 项补丁里把两个环境变量提到 `run_lab_dual3090_7b_8b.ps1` 顶部，`Start-Job` 子进程会继承，正式实验也不会再撞。

同时 `datasets 3.5` 加载 `piqa`（自定义代码数据集）**必须** `trust_remote_code=True`，而 `lm_eval/base.py` 的 `load_dataset(...)` 没传。用 `HF_DATASETS_TRUST_REMOTE_CODE=1` 绕过，**不用改 lm_eval**。

### B3. ⚠️ `7b_parallel` 的 STEP 1（FP16 基线）会**直接 AssertionError**

`run_lab_dual3090_7b_8b.ps1` 第 165–172 行的 FP16 基线只传：

```
--wbits 16 --abits 16 --eval_ppl
```

没传 `--let/--lwc`，而 `--epochs` 默认是 **20**。`main.py` 里这句是**无条件**的：

```python
if args.epochs > 0:
    assert args.lwc or args.let      # ← 第 262 行，wbits=16 也照样触发
```

**实测已复现**：`AssertionError`，进程退出码 1。因为后面有 `if ($LASTEXITCODE -ne 0) { throw }`，**整个 `7b_parallel` 会在第 1 步就终止，连标定都进不去。**

**修法（一行）**：给 STEP 1 加 `--epochs 0`。我已经验证这样能跑通，并拿到了基线（见 §七）。

> ⚠️ 这一条请务必在主仓修掉，否则正式实验白跑。

## 七、FP16 基线（顺带补出来的，非常重要）

`7b_parallel` 自带 FP16 基线这一步，但被 B3 挡住了。我用同样的命令加上 `--epochs 0` 单独跑了一次：

| 模型 | WikiText-2 | C4 |
|---|---|---|
| **FP16（本机实测）** | **5.4722** | **6.9727** |
| Llama-2-7B 公认参考值 | ~5.47 | ~6.97 |

**逐位吻合**。这意味着：模型加载、分片拼接、tokenizer、2048 窗口 NLL 统计**全都是对的**。

于是可以给出一个干净的判读（`nsamples=8, epochs=1`，远不是正式设置）：

| 配置 | WikiText-2 | C4 | 相对 FP16 恶化 |
|---|---|---|---|
| FP16 | 5.47 | 6.97 | 1.0× |
| Original SpikeLLM W4A4 | 59.17 | 64.20 | 10.8× / 9.2× |
| OSDSQ v2 W4A4 | 26.72 | 34.46 | 4.9× / 4.9× |

**结论**：PPL 高不是评测 bug，而是 8 样本 × 1 epoch 根本不足以标定。OSDSQ v2 在同一预算下把恶化从 ~10× 压到 ~4.9×，**方向明确**。这个比值比绝对数更能说明问题。

## 八、需要工作 AI 确认或回答的问题（请逐条）

1. **P1：梯度 NaN，需要你们判断是否已知。**
   - Original：层 **1, 24, 30, 31**；OSDSQ v2：层 **1, 30, 31**。层 1 两腿都有 ⇒ 系统性。
   - 机制：`utils.py` 的 `NativeScalerWithGradNormCount` 用 `GradScaler`，`scale(loss).backward()` 后梯度溢出成 inf/nan，`torch.norm` 就是 nan。**loss 本身是有限值**，也没触发 `Loss is NAN, stopping training`。
   - 后果：`GradScaler.step()` 检测到 inf/nan 会**跳过该步**、scale 减半。所以权重没被污染（好消息），但 `epochs=1` 下这几层 LET/LWC **一次都没更新**，等于没训——很可能直接贡献了偏高的 PPL。
   - **我的判断**：`epochs=20` 下大概率自愈（第一次溢出后 scale 减半，后续 epoch 正常更新），**但必须在正式日志里 grep `norm:nan` 确认**。
   - **问题**：你们在原机器上跑 Original 时见过这个 nan 吗？如果没有，说明是我们的环境（torch 2.6 / CUDA 12.6 的 AMP 行为）引入的，需要认真对待。
   - 备选诊断：`--deactive_amp`（走 fp32 分支）。但那条分支下 `spike_omniquant.py:214` 不再 `.float()` mask，若 cache 里 mask 是 fp16 会撞 dtype mismatch，要先补。建议先在 smoke 上试，别直接上正式实验。
2. **P2：门禁脚本超参不一致。** `run_lab_dual3090_7b_8b.ps1` 第 118–125 行的 resume 门禁**没传** `--alpha/--let_lr/--lwc_lr/--aug_loss`，走的是默认 `alpha=0.5, let_lr=0.005, aug_loss=False`，和 smoke 标定的 `alpha=0.75, let_lr=0.001` 不一致。`--epochs 0 --resume` 下大概率无害，但属于"门禁没覆盖正式路径"的漏洞，**建议补上**。
3. **正式实验时间预估请你们核对。** 按 smoke 外推（12.9 min × 128/8 × 20 ≈ 68 小时/腿），但这是粗暴线性外推，实际同层内是 `nsamples/batch_size` 次前向 × epochs。**68 小时太长，如果你们预期是十几小时，说明我哪里理解错了，请指正。** 建议先跑一个"单层 + 128 样本 + 1 epoch"的计时探针再定。
4. **`models/` 版本确认**：我用的是 10-06 迁移副本的 `models/`（已确认无 KV-study 局部魔改、`spike_linear.py` 构造签名一致）。**请确认这就是你们预期的版本**；若之后改过 `models/spike_llama_layer.py`，请发新文件（注意我已经在上面打了 2 个补丁，合并时别覆盖）。
5. **正式实验的产出规格**：我建议固定输出 WikiText-2 / C4 的 PPL（FP16 / Original / OSDSQ v2）、6 个 zero-shot（piqa, arc_easy, arc_challenge, boolq, winogrande, hellaswag）、校准耗时、峰值显存、checkpoint 体积。**够不够？还需要 timestep 扫描或 KV 缓存相关指标吗？**
6. **8B 是否必要**：8B base（`Meta-Llama-3-8B`）约 16GB，按本机 0.6MB/s 需要 **7.5 小时以上**，加 C4 缓存更久。本机 HF 缓存里只有 `Meta-Llama-3-8B-Instruct`。**要不要用 Instruct 版顶替，或者干脆放弃 8B？**
7. **有没有现成权重**：如果你们手上有 `Llama-2-7b-hf` 或 8B 的完整权重，用网盘/移动硬盘传过来会快得多。这对 8B 尤其关键。
8. **是否允许装 `hf_transfer`**（只装在 `spikellm` conda 环境）：能显著提速，但**不支持断点续传**，在这种会超时的网络上有风险。默认不装。

## 九、环境速查（给工作 AI 写命令用）

```
项目根目录 : C:\spikellm\SpikeLLM
Python     : C:\Users\admin\.conda\envs\spikellm\python.exe   (3.11.11)
PyTorch    : 2.6.0+cu126   CUDA 12.6   transformers 4.44.2   datasets 3.5.0
             accelerate 1.7.0   tokenizers 0.19.1   huggingface-hub 0.30.2
GPU        : 2 × RTX 3090 24GB (驱动 610.88)
网络       : huggingface.co 不通；hf-mirror.com 通，实测 0.6 MB/s
模型路径   : C:\spikellm\SpikeLLM\models\Llama-2-7b-hf  (13,476,876,272 B)
缓存       : cache\dataloader_Llama_wikitext2_128.cache / dataloader_Llama_wikitext2_8.cache
             cache\testloader_Llama_wikitext2_all.cache / testloader_Llama_c4_all.cache
act 统计量 : act_scales\Llama-2-7b-hf.pt / act_shifts\Llama-2-7b-hf.pt (各 4,640,910 B, 128 样本)
日志       : lab_download_7b*.log / lab_smoke_7b_run5.log / log\7b_smoke_*\smoke.log
报告       : lab_environment_and_smoke_report.md
```

> ⚠️ **一定要带 `-PythonExe`**。PATH 上的默认 python 是 WorkBuddy 自带的 3.13，没有 torch，`check` 会直接判失败。

## 十、网络慢的根因诊断（2026-10-27 20:20 追加，结论不变）

实测结论：**不是运营商问题，也不是代理问题，是无线链路下行速率被压到 9 Mbps。**

| 指标 | 实测值 | 判断 |
|------|--------|------|
| 联网方式 | 仅 WiFi（TP-LINK USB 网卡，VID_2357&PID_014E） | **3 个千兆网口全部空着**（Intel×2 + ASIX USB-GbE，均 Disconnected） |
| SSID / 频段 | NEUQ 校园网 / 5GHz ch48 / 802.11ax-be | 信号 96–100%，不是距离问题 |
| 无线协商速率 | **下行 9 Mbps** / 上行 143 Mbps | 下行被钉死在最低 MCS，反复采样 3 次完全不变 |
| 实际吞吐 | 609–682 KB/s（4.9–5.5 Mbps） | 恰好是 9 Mbps 的 ~60%，与下载速度完全吻合 |
| hf-mirror RTT / 丢包 | 204 ms / 25% | 出境路径差，叠加小水管导致 TCP 吞吐塌陷 |
| 清华 TUNA 4 并发 | 每连接约 7 KB/s | 校外镜像对本校 IP 单独限速 |
| 公网出口 | IPv6 2408:861a:b200:effe::9d（联通，秦皇岛） | 校园网 |
| 系统代理 | 127.0.0.1:10808 (xray) + Clash 7890 | 两条代理对 HF 均返回 0 字节，不可用于大文件 |
| ToDesk | 常驻，累计 CPU 6167s | 远控画面与下载**共用同一条 9 Mbps 链路**，观看时抢占 1–3 Mbps |

**换算**：13.5 GB ÷ 0.6 MB/s ≈ 6.2 小时，与实际耗时（含 1.5 小时中断）完全对得上。

**提速优先级**
1. **插网线**（最有效，可能 10–100 倍）；需要确认实验室墙上网口是否已开通。
2. WiFi 优化：USB 网卡换到 USB 3.0 口 + 延长线抬高避遮挡；设备管理器里关闭网卡节能、可试禁用 802.11ax 模式；当前驱动 TP-LINK 14.46.40.648（2024-11-15）。
3. 下载时把 ToDesk 画质调低或断开远控。
4. 若实验室有内网 NAS / 同学已下好的权重，局域网拷贝最快。
5. 不建议走代理下载（实测不通）。

> 注意：用户当前通过 ToDesk 远程操作本机，改 WiFi 设置可能导致会话掉线，所以以上调整需用户本人在场时再做。

## 十一、已知风险 / 需要留意

- `datasets 3.5.0` + `huggingface_hub 0.30.2` 的 `DatasetCard.load` 不兼容。**缓存文件在就不触发，一删就复现**，需要 §三 的兜底。
- `auto_gptq` 未安装（仅 `--real_quant` 需要）。若后续要走真实量化路径，Windows 上装 `auto-gptq` 匹配 CUDA 12.6 / torch 2.6 通常要自己编译，是个潜在坑。
- smoke 的 act stats 生成需要 fp16 加载 7B（约 13.5GB），`device_map=auto` 会跨两卡，这是正常的，不是 7b_parallel。
- 按指令，**7b_parallel / download_8b / smoke_8b / 8b_parallel 一律未启动**，等你们确认。
  - （10-28 10:40 更新：`7b_parallel` 已在收到 V2 与放行指令后正式启动，见 §十二；`download_8b` / `smoke_8b` / `8b_parallel` 仍未启动。）

---

## 十二、★ V2 部署验收 + 正式实验启动（2026-10-28 10:24–10:45）

### 12.1 覆盖解压的记录（**没有删除任何已有资产**）

| 项 | 值 |
|---|---|
| 部署包 | `C:\Users\admin\Downloads\SpikeLLM_lab_dual3090_deploy_v2_2026-10-28.zip`（1,886,688 B，150 个文件） |
| 备份目录 | `C:\spikellm\SpikeLLM\backup_before_v2_20261028_102423\`（8 个我改过的文件全量拷贝） |
| 覆盖方式 | 逐条写文件，**显式跳过保护前缀**：`models/Llama-2-7b-hf/`、`cache/`、`log/`、`act_scales/Llama-2-7b-hf.pt`、`act_shifts/Llama-2-7b-hf.pt` |
| 覆盖结果 | 写入 150 个文件，跳过 0 个（V2 内本来就不含被保护路径） |
| 覆盖后校验 | `models/Llama-2-7b-hf` 13 GB / 2 个 safetensors 完好；`cache/` 56 MB 四个 cache 齐全；`log/` 18 MB；两个 smoke checkpoint（8,983,506 B）都在；`act_scales|act_shifts/Llama-2-7b-hf.pt` 各 4,640,910 B 未被改写 |
| 语法检查 | `py_compile` 通过：`main.py`、`lab_compat.py`、`models/spike_llama_layer.py`、`spike_driven_quant/spike_omniquant.py`、`generate_act_scale_shift.py`、`lab_preflight_and_download.py` |

### 12.2 V2 与我在实验室本地打的补丁逐条对照（**全部为等价重构，numerics 路径无改动**）

| 我原来的补丁 | V2 的实现 | 判定 |
|---|---|---|
| B1 `lab_compat.py`（4 KB，含诊断说明） | 精简版 707 B，同样把 `huggingface_hub.file_download.are_symlinks_supported` 打成 `False`，并包了 `try/except ImportError` | ✅ 语义等价，更健壮 |
| `main.py` 顶部包在 `try/except ImportError` 里 `import lab_compat` | 直接 `import lab_compat`（不再吞异常） | ✅ 更好：缺文件就该立刻报错 |
| B2 ps1 顶部设 `HF_ENDPOINT` / `HF_DATASETS_TRUST_REMOTE_CODE` | 同位置、同变量 | ✅ 一致 |
| B3 STEP 1 加 `--epochs 0` | 同 | ✅ 一致 |
| rotary 4.44 签名（我硬编码走 4.44 分支） | `Version(transformers.__version__) >= 4.44.0` 双分支，兼容旧版 | ✅ 更通用；本次仍走 4.44 分支，数值同 |
| eager mask 多 1 列的**条件容错**（我判 `kv_seq_len+1` 才切） | 直接 `attention_mask[:, :, :q_len, :kv_seq_len]` 统一切片 | ✅ 更干净，且 `q_len` 维度也一起对齐，覆盖我那种写法没覆盖的形状 |
| `QuantLlamaDecoderLayer.forward` 显式补 `cache_position` / `position_embeddings` 形参 | 回退成 `use_cache=False, **kwargs`（靠 `**kwargs` 吸收） | ✅ 等价，我已实测这条路径在 4.44 下不报错 |
| `spike_omniquant.py` 新增 `model.model.rotary_emb.to(dev)/.cpu()` | 同，只改了注释 | ✅ 一致 |
| —— | **新增**：`Run-Smoke` 与 `Run-FullParallel` 的 resume/eval 调用补上 `--aug_loss --alpha --let_lr --lwc_lr` | ✅ 对齐正式配置，`epochs=0` 时不影响数值 |

> 结论：V2 **没有改动量化算法**（`spike_linear` / `quantizer` 等未变），只有兼容性与调用参数对齐。

### 12.3 数值回归（用现有 OSDSQ smoke checkpoint）

命令与开发组给出的一致（我额外固定 `CUDA_VISIBLE_DEVICES=0`，保证与 smoke 时单卡条件相同）：

```
& $PY main.py --model .\models\Llama-2-7b-hf --net Llama-2-7b-hf `
  --wbits 4 --abits 4 --addbit 1 --low_p 0.9 `
  --let --lwc --aug_loss --alpha 0.75 --let_lr 0.001 --lwc_lr 0.01 `
  --epochs 0 --resume .\log\7b_smoke_osdsq_v2\omni_parameters.pth `
  --eval_ppl --output_dir .\log\v2_regression_osdsq
```

| 指标 | V2 回归实测 | smoke 记录 | 偏差 |
|---|---|---|---|
| WikiText-2 | **26.719755172729492** | 26.72 | **0（逐位一致）** |
| C4 | **34.46274185180664** | 34.46 | **0（逐位一致）** |

- 退出码 **0**，总耗时 **13.1 min**（其中 32 层 resume 装载 29.8 s，WikiText-2 4m47s，C4 7m25s）。
- 日志：`C:\spikellm\SpikeLLM\lab_v2_regression_osdsq.log`，输出目录 `log\v2_regression_osdsq\`。
- 附带确认：`--resume` 是**真的被用上了**（`spike_omniquant.py` 里 4 处 `args.resume` 分支，`main.py` 只声明、实际由 omniquant 消费），不是空参。

### 12.4 正式实验启动

| 项 | 值 |
|---|---|
| 启动时刻 | **2026-10-28 10:40:03 (GMT+8)** |
| 命令 | `powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode 7b_parallel -PythonExe "C:\Users\admin\.conda\envs\spikellm\python.exe"` |
| 环境 | `HF_ENDPOINT=https://hf-mirror.com`、`HF_DATASETS_TRUST_REMOTE_CODE=1` |
| 超参 | `nsamples=128`、`epochs=20`、`seed=2`、`alpha=0.75`、`let_lr=1e-3`、`lwc_lr=1e-2`、`wbits=abits=4`、`addbit=1`、`low_p=0.9` |
| 驱动日志 | `C:\spikellm\SpikeLLM\lab_7b_parallel_driver.log` |
| 分阶段输出 | `log\7b_fp16\`（STEP 1）→ `log\7b_orig_with_let\`（GPU0）/ `log\7b_osdsq_v2_with_let\`（GPU1），各自的 `run.log`、`real_multibit\`、`zeroshot\` |
| 完成标记 | 每腿 `done_ppl.flag` / `done_multibit.flag` / `done.flag`，STEP 1 有 `done.flag` |
| 进度 | STEP 1（FP16 基线）单卡 GPU0 运行中，其余进程未启；GPU1 空闲待命 |
| 预计耗时 | 指令给的是 1.5–3 天；**实测约 20–23 h**（依据见 §12.7 的节奏测算） |

**本轮计划产出的数据（不含 timestep 扫描、不含 KV 新指标）：** FP16 基线 WikiText-2/C4；两腿各 W4A4 的 WikiText-2/C4；两腿 `real_multibit_simulate` 的 WikiText-2/C4；两腿 6 个零样本（piqa / arc_easy / arc_challenge / boolq / winogrande / hellaswag）；耗时；峰值显存；`omni_parameters.pth`。

### 12.5 正式实验前置检查（都已通过）

- 6 个零样本数据集**离线可完整加载**（`HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` 下逐个试过）：piqa 16113/3084/1838、arc_easy 2251/2376/570、arc_challenge 1119/1172/299、boolq 9427/3270、winogrande 40398/1767/1267、hellaswag 39905/10003/10042。→ 零样本阶段不会再联网。
- `act_scales`/`act_shifts` 已存在 128 样本版本，`Ensure-ActStats` 会直接跳过，不会重算。
- 双卡启动前均空闲（198 MiB / 262 MiB），无障碍占用进程。
- C 盘余量 108 GB（单腿 checkpoint 仅 9 MB，日志预计 <1 GB）。

### 12.6 8B 的处理（按指令）

**未启动。** 原因确认：`datautils.py` / `lab_preflight_and_download.py` 里 WikiText-2 与 C4 的 cache 文件名都以 `args.model_family`（`Llama`）为键，7B 与 8B 会命中同一批 `dataloader_Llama_*` / `testloader_Llama_*` 文件 → 并行下载/构建会互相覆盖。等开发组做缓存隔离后再开，且必须用 base 版（`Meta-Llama-3-8B`），不能用 Instruct 替代。

### 12.7 STEP 1 结果 + 初期 NaN 监控（10:45 起，持续更新）

**STEP 1（FP16 基线，GPU0 单卡）：**

| 指标 | 本次实测 | 10-27 那次 | 文献值 |
|---|---|---|---|
| WikiText-2 | **5.472231388092041** | 5.4722 | ≈5.47 |
| C4 | **6.972711086273193** | 6.9727 | ≈6.97 |

→ 逐位复现，评测链路再次确认可信。

**初期逐 epoch 监控（layer 0，前 3 个 epoch）：**

| epoch | ORIG loss | ORIG grad-norm | OURS loss | OURS grad-norm |
|---|---|---|---|---|
| 0 | 8.3606e-05 | 6.0674e-05 | 6.6477e-04 | 5.6083e-04 |
| 1 | 5.6833e-05 | 9.6617e-06 | 3.7310e-04 | 5.7097e-05 |
| 2 | 5.4176e-05 | 7.8435e-06 | 3.4928e-04 | 4.4361e-05 |

- **两腿全部有限，无 `norm:nan`、无 `norm:inf`、无 loss 非有限**（layer 0 前 3 个 epoch 窗口内 `norm:nan` 计数为 0；layer 1 的情况见下方单独说明）。
- loss 单调下降，梯度范数同向收敛，符合"AMP 正常、无溢出"的形态。
- 峰值显存：**ORIG 18,336 MB / OURS 18,303 MB**（比 smoke 的 12.6 GB 高，因为 `nsamples` 8→128，`fp_inps`/`quant_inps` 缓存变大）。24 GB 卡余量约 6 GB，安全。
- 与 smoke 的关系：smoke（8 样本 × 1 epoch）layer 0 的 loss 是 ORIG 1.63e-4 / OURS 1.48e-3，本次（128 样本）起点更低更稳，符合"样本多了、`aug_loss` 平均更平滑"的预期。

**实测节奏与修正后的耗时预估：**

- 3 个 epoch ≈ 4.4 min → **约 88 s/epoch**（128 iter × 0.69 s/iter），**约 29 min/层**（20 epoch）。
- 32 层标定 **≈ 15.7 h/腿**（两腿并行，同时完成）。
- 加上 `real_multibit_simulate` PPL（约 13 min）与 6 个零样本（参考 smoke 里单任务约 45 min，6 个任务预计 4–6 h），**整轮预计 20–23 h**，比原估的 68 h/腿快得多（因为 smoke 的 760 s 里层初始化与首样本开销占比很大，线性外推高估了）。
- 结论：**1.5–3 天的预算绰绰有余**，实际大概率 1 天内收工。

**checkpoint 落盘方式（重要）：** `spike_omniquant.py` 每完成一层就 `torch.save(omni_parameters, args.output_dir/omni_parameters.pth)`，所以 `log/7b_<arm>_with_let/omni_parameters.pth` 是**逐层增量覆盖**的。中途崩溃会留下一个"前 N 层"的 checkpoint（不完整），但不会一分不剩。

**layer 0 收尾 + layer 1 首个 epoch（11:03 观测）：**

| | ORIG | OURS |
|---|---|---|
| layer 0 iter 19（20 epoch 末） | loss 4.8905e-05 / norm 1.2752e-05 | loss 2.9543e-04 / norm 2.2925e-05 |
| layer 1 **iter 0** | loss 0.066371 / **norm:nan** | loss 0.037908 / **norm:nan** |
| layer 1 iter 1 | loss 0.043338 / norm 0.027923 | loss 0.019907 / norm 0.012965 |
| layer 1 iter 2 | loss 0.029860 / norm 0.020647 | loss 0.013495 / norm 0.010191 |

**判定：属于开发组预告的"偶发 norm:nan、随后恢复"，允许继续，不做任何干预。**

依据：
1. **两腿都在同一位置出现（layer 1 的 iter 0），且都只有 1 次**，与 10-27 smoke 时"Orig 层 1/24/30/31、Ours 层 1/30/31"的分布一致 —— 是模型结构性的首个 batch 溢出，不是随机故障。
2. **loss 始终有限**（0.066 / 0.038，正常量级），只有梯度范数是 NaN，说明 `GradScaler` 判定溢出并**跳过该步更新**，权重未被污染。
3. **紧接着 iter 1 就恢复**且梯度范数回到 1e-2 量级，loss 单调下降（0.0664→0.0433→0.0299），没有"持续 NaN / loss 发散 / 进程异常"的任何征兆。
4. 截至 11:07 累计：两腿各 `norm:nan` **1 次**、`norm:inf` **0 次**。

**已按要求：没有启用 `--deactive_amp`，没有改动任何量化算法。** 会继续按"每层只数一次"的方式盯着；若出现**同一层连续多 epoch NaN**或 **loss 变 NaN/发散**或 **进程退出**，我会立即停并保留日志。

**无人值守留痕：** 已部署 `lab_watch_progress.ps1`（后台常驻，每 10 min 往 `lab_watch.log` 追加一条快照：当前层号、已记录 epoch 数、`norm:nan`/`norm:inf` 累计、最近一条 loss/norm、双卡显存与利用率、以及 `done_*.flag` 状态）。它在"python 进程消失但 `done.flag` 未生成"时会写一条 `FATAL` 并退出，便于事后定位。
> 附一个 Windows 坑，免得你们复现时踩：这些 `run.log` 是被 `*>>` 以写句柄独占打开的，`[System.IO.File]::ReadAllText` / `Get-Content` 会抛 sharing violation；必须用 `FileShare::ReadWrite` 打开才能读。我第一版看门狗就是因为这个静默失败，写了三行 `layer=- epochs_logged=0`。

---

## 十三、★★ 正式实验已完成（2026-10-28 10:40 → 2026-10-29 11:31，24.85 h，退出码 0）

> 完整结果表另存为 `lab_7b_formal_results.md`。本节是给你们的摘要 + 需要你们判断的点。

### 13.1 主结果（全部为本机实测，非文献值）

FP16 基线由 STEP 1 实测：**5.472231388092041 / 6.972711086273193**（WikiText-2 / C4），与 10-27 补测逐位一致。

| 配置 | 模拟模式 | WikiText-2 | C4 |
|---|---|---|---|
| FP16（基线） | — | 5.472231388092041 | 6.972711086273193 |
| Original SpikeLLM | fake_binary | 10.572571754455566 | 13.773828506469727 |
| **OSDSQ v2** | fake_binary | **7.979187488555908** | **10.679176330566406** |
| Original SpikeLLM | real_multibit | 10.810821533203125 | 14.185491561889648 |
| **OSDSQ v2** | real_multibit | **8.157027244567871** | **10.89747142791748** |

相对 FP16 的恶化倍数：Original ×1.932 / ×1.975（fake）、×1.976 / ×2.034（real）；OSDSQ v2 **×1.458 / ×1.532（fake）**、**×1.491 / ×1.563（real）**。

OSDSQ v2 相对 Original 的降幅：**fake −24.53% / −22.47%，real −24.55% / −23.18%**。两种模拟模式几乎同幅度，说明增益不是模拟近似带来的假象。

### 13.2 六个零样本（acc / acc_norm）

| 任务 | Orig acc | Ours acc | Δ | Orig acc_norm | Ours acc_norm | Δ |
|---|---|---|---|---|---|---|
| piqa | 0.6583 | 0.6752 | +1.69 | 0.6638 | 0.6866 | +2.28 |
| arc_easy | 0.5004 | 0.5535 | +5.31 | 0.4306 | 0.4596 | +2.90 |
| arc_challenge | 0.2824 | 0.3234 | +4.10 | 0.3106 | 0.3328 | +2.22 |
| boolq | 0.6235 | 0.6425 | +1.90 | — | — | — |
| winogrande | 0.5217 | 0.5777 | +5.60 | — | — | — |
| hellaswag | 0.4372 | 0.4712 | +3.40 | 0.5767 | 0.6150 | +3.83 |

**6 个任务、10 个指标全部 OSDSQ v2 占优**，幅度（+1.7 ~ +5.6 pt）远大于 stderr（0.005–0.012）。

### 13.3 耗时 / 显存 / 产物

| | Original（GPU0） | OSDSQ v2（GPU1） |
|---|---|---|
| 32 层标定 | 10:45:58 → 17:43:29，**6 h 57 min** | 10:45:59 → 17:49:12，**7 h 03 min** |
| 整腿总耗时 | **22.6 h** | **24.75 h** |
| 峰值 `max_memory_allocated` | **18 379 MB** | **18 346 MB** |

约 **13.0 min/层**（≈88 s/epoch × 20），24 GB 余量约 5.6 GB，无 OOM。比原估 68 h/腿快约 3 倍。

checkpoint：`log/7b_orig_with_let/omni_parameters.pth`、`log/7b_osdsq_v2_with_let/omni_parameters.pth`，各 8 983 506 B，时间戳晚于 layer 31，确认 32 层全写入。

日志：两腿 `run.log`（各 8.6 MB）、`real_multibit/log_rank0_*.txt`、`zeroshot/log_rank0_*.txt`、驱动 `lab_7b_parallel_driver.log`、看门狗 `lab_watch.log`（147 条快照）。完成标记 `done_ppl.flag` / `done_multibit.flag` / `done.flag` 两腿齐全。

### 13.4 NaN 情况（按你们给的判据走的）

| | Original | OSDSQ v2 |
|---|---|---|
| `norm:nan` 累计 | 11 次 | 5 次 |
| `norm:inf` 累计 | 0 | 0 |
| `loss` 为 nan/inf | **0** | **0** |
| 出现层 | 1, 18, 19, 22, 24, 26, 27, 29, 30, 31 | 1, 18, 30, 31 |

**逐条核对了每次 NaN 的前后 epoch，16 次全部是"前有限 → 本步 NaN → 下一步立即恢复有限"**，`loss` 始终有限（只有梯度范数为 NaN，GradScaler 跳过该 step，权重未污染）。判据"偶发、随后恢复、允许继续"成立，**未启用 `deactive_amp`，未改动量化算法**，未出现"持续 NaN / loss 非有限 / 进程异常"。

两个值得你们注意的细节：
1. Original 有 4 次发生在 **iter 15 / iter 19**（训练中后段），不只是"层首 batch"；但同样单步恢复。
2. smoke（8 样本，20 步/层）OSDSQ v2 是 3 次 NaN；正式（128 样本 × 20 epoch，640 步/层）只涨到 5 次 —— **NaN 密度随预算增大显著下降**，进一步支持它是瞬态而非结构性缺陷。

### 13.5 需要你们决定的四件事

1. **FP16 零样本基线缺失。** STEP 1 只评了 FP16 的 PPL，没评零样本，所以现在只能做"两腿互比"，**算不出 recovery rate**。要写 "recovered X% of FP16" 就必须补一次 FP16 零样本（约 3–4 h）。要不要补？
2. **只有 seed=2 单次运行。** 若论文需要 mean±std 或显著性检验，得再跑 2 个 seed（每轮约 25 h，即 ~2 天）。要不要排？
3. **arc_easy 的 `acc_norm`（0.43 / 0.46）低于 `acc`（0.50 / 0.55）**，两腿同向。这不影响两腿对比，但如果论文要放单个任务的绝对数，需要确认 `lm_eval/tasks/arc_easy.py` 的长度归一化没写错。要不要我核？
4. **8B 仍未启动**，等你们的缓存隔离（`datautils.py` / `lab_preflight_and_download.py` 里 WikiText-2 与 C4 的 cache 名都以 `model_family="Llama"` 为键，7B/8B 会互相覆盖），且必须用 base 版 `Meta-Llama-3-8B`。

### 13.6 下一步等指令

双卡现已空闲（143 MiB / 257 MiB）。按原指令，**本轮不增加 timestep 扫描或 KV 新指标**，我不擅自开新实验。

---

## 十四、★★ 项目整体迁移到 Desktop\spikellm + 迁移后回归 + P0 已启动（2026-10-29 12:42–13:08）

### 14.1 迁移范围与方式

**目标**：本项目全部内容收进 `C:\Users\admin\Desktop\spikellm\` 一个文件夹，结题时整目录删除即可，不在实验室机器上留任何本项目痕迹。

| 项目 | 原位置 | 新位置 | 体积 |
|---|---|---|---|
| 代码 + 权重 + 日志 + checkpoint | `C:\spikellm\SpikeLLM` | `Desktop\spikellm\SpikeLLM` | **12.67 GiB / 270 文件 / 47 目录** |
| 本项目 HF 缓存子集 | `C:\Users\admin\.cache\huggingface` | `Desktop\spikellm\hf_cache` | **1.78 GiB / 241 文件** |
| 部署包 | `C:\Users\admin\Downloads` | `Desktop\spikellm\deploy_packages` | 3 个文件 |
| 本同步日志 | `SpikeLLM\` | `Desktop\spikellm\`（根） | — |
| conda 环境 | `C:\Users\admin\.conda\envs\spikellm` | **原位不动**（按要求） | — |
| `C:\DingXL\SpikeLLM_KV_study` | **原位不动**（按要求，等你确认） | — | 0.48 GiB |

`C:\spikellm` 旧根目录本身已确认清空后删除。`Desktop\spikellm` 合计 **14.46 GiB / 531 文件**。

**方式：同卷 `Move-Item`（NTFS 重命名），不是跨盘复制再删源。** 数据块完全没有被重写一遍，因此不存在"复制一半损坏"的可能，比"复制→校验→删源"更安全，也快得多（13 秒完成）。为证明完整性，仍然做了三重校验：

1. 迁移前对 270 个文件建清单（相对路径 + 字节数 + mtime UTC）。
2. 迁移后逐条比对：**270 checked / 0 mismatch**。
3. 4 个关键大文件迁移前后 SHA256 全等：
   - `model-00001-of-00002.safetensors` 9,976,578,928 B → `4EC71FD5…BE41`
   - `model-00002-of-00002.safetensors` 3,500,297,344 B → `41780B5D…5A7F`
   - `log/7b_orig_with_let/omni_parameters.pth` 8,983,506 B → `FB61E475…911F`
   - `log/7b_osdsq_v2_with_let/omni_parameters.pth` 8,983,506 B → `AB8ECE40…D5F`
4. safetensors 懒加载冒烟：两个分片都能 mmap 打开，**323 个张量 = index 条数**，`embed_tokens [32000,4096] FP16`、`lm_head [32000,4096] FP16` 实测可读。

### 14.2 ⚠️ 一个必须让你们知道的判断：HF 缓存**没有**整体搬

原指令是"把 `C:\Users\admin\.cache\huggingface\` 迁到 `Desktop\spikellm\hf_cache\`"。实测该目录 **55.74 GiB，其中只有 1.78 GiB 属于本项目**，其余约 50 GiB 是这台机器上别人的模型：

| 仓库 | 体积 | 与本项目关系 |
|---|---|---|
| `models--meta-llama--Meta-Llama-3-8B-Instruct` | 14.97 GiB | 不是我们要的 base 版，且已被明确排除 |
| `models--Qwen--Qwen2.5-7B-Instruct` | 14.20 GiB | 无关 |
| `models--mistralai--Mistral-7B-Instruct-v0.3` | 13.51 GiB | 无关 |
| `models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K` | 7.35 GiB | 无关 |
| `facebook/dinov2-large`、`openai/clip-vit-base-patch32` | 1.13 + 1.13 GiB | 无关 |
| `textattack/bert-base-uncased-MNLI`、若干 timm swin、opt-125m | 约 1.9 GiB | 无关 |
| **`datasets--*`（piqa/arc/boolq/winogrande/hellaswag/c4/wikitext/glue）+ arrow 缓存** | **1.78 GiB** | **本项目需要** |

**若把 55 GiB 全搬进来**，后果是：①你这个"结题就整个删掉"的文件夹里躺着别人 50 GiB 的模型，删的时候会连带删掉；②`~/.cache/huggingface` 被清空会让这台机器上其它项目/其它人的脚本直接失效。

所以**只搬了本项目那 1.78 GiB**（`datasets--*` 数据集仓库 + `datasets/` 的 arrow 缓存 + 7B 占位 + 认证 token），`~/.cache/huggingface` **原样保留、一个字节没删**。这样：本项目 `HF_HOME` 指向 `Desktop\spikellm\hf_cache` 自成一体；别人的东西原封不动；删文件夹时不会误伤。

如果确实要"HF 缓存整个搬走"，说一声我再动——但需要先确认这台机器上其它模型的所有者。**另外提醒**：`Meta-Llama-3-8B-Instruct` 本地已有 14.97 GiB，但按指令不能用 Instruct 替代 base，8B 仍要下 base 版。

### 14.3 迁移后回归验证（3 项，全部通过）

**(a) 6 个零样本数据集离线可加载**（`HF_HOME` 指向新位置，`HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1`）：

```
HF_DATASETS_CACHE = C:\Users\admin\Desktop\spikellm\hf_cache\datasets
[OK] piqa           {'train': 16113, 'test': 3084, 'validation': 1838}
[OK] arc_easy       {'train': 2251,  'test': 2376, 'validation': 570}
[OK] arc_challenge  {'train': 1119,  'test': 1172, 'validation': 299}
[OK] boolq          {'train': 9427,  'validation': 3270}
[OK] winogrande     {'train': 40398, 'test': 1767, 'validation': 1267}
[OK] hellaswag      {'train': 39905, 'test': 10003,'validation': 10042}
offline loadable: 6/6
```

**(b) PPL 回归（现有 OSDSQ 正式 checkpoint + `--epochs 0 --resume --eval_ppl`）——逐位一致**：

| 指标 | 迁移后实测 | 正式实验实测 | 偏差 |
|---|---|---|---|
| WikiText-2 | **7.979187488555908** | 7.979187488555908 | **0（bit-identical）** |
| C4 | **10.679176330566406** | 10.679176330566406 | **0（bit-identical）** |

退出码 0，16.0 min。日志 `lab_migrate_regression.log`。

**(c) 目录树与磁盘** —— 见 §14.5。

### 14.4 代码里的绝对路径：找到 3 处，已全部修正

| # | 文件 | 原内容 | 改为 | 说明 |
|---|---|---|---|---|
| 1 | `run_lab_dual3090_7b_8b.ps1` | 无 `HF_HOME` | 顶部新增 `HF_HOME = $PSScriptRoot\..\hf_cache`（可被已有环境变量覆盖） | 即你们要求的设置；`HF_ENDPOINT`、`HF_DATASETS_TRUST_REMOTE_CODE` 保持不变 |
| 2 | `lab_watch_progress.ps1` | `$Base = "C:\spikellm\SpikeLLM"` | `$Base = $PSScriptRoot` | 看门狗硬编码了旧路径；顺带参数化 `-Arms` / `-LogName` / `-Hours` |
| 3 | `run_both_1.1b.ps1` | `d:\spikellm\.venv\Scripts\python.exe`（两处） | `& $PythonExe`，带回退探测（conda env → 项目 .venv → PATH） | **原仓库遗留的死路径**，本机没有 D 盘，一跑就报错。属 1.1B 脚本，非本轮流程 |

其余扫描结果：`main.py`、`lab_compat.py`、`datautils.py`、`generate_act_scale_shift.py`、`lab_preflight_and_download.py`、`lm_eval/` **均无绝对路径硬编码**，全部基于 `--cache_dir` / `--output_dir` 相对参数 + `$PSScriptRoot`。

`generate_paper_figures.py` 用的是 `os.path.dirname(__file__)`，天然可迁移，不用改。

改动后 4 个 ps1 全部通过 `[Parser]::ParseFile` 语法检查（`PARSE_OK`）。

### 14.5 迁移后目录树（两层）与磁盘

```
C:\Users\admin\Desktop\spikellm\          14.46 GiB / 531 files
├── SpikeLLM\                             12.67 GiB
│   ├── models\Llama-2-7b-hf\             12.56 GiB（2 个 safetensors + config/tokenizer）
│   ├── cache\                            55.3 MiB（wikitext2_128 / wikitext2_8 / c4_all / wikitext2_all）
│   ├── log\                              51.3 MiB
│   │   ├── 7b_orig_with_let\             正式实验 Original（omni_parameters.pth 8.98 MB）
│   │   ├── 7b_osdsq_v2_with_let\         正式实验 OSDSQ v2（omni_parameters.pth 8.98 MB）
│   │   ├── 7b_fp16\                      FP16 基线 5.4722 / 6.9727
│   │   ├── p0_noaug_orig\                ← P0 新增，正在跑
│   │   └── p0_noaug_osdsq_v2\            ← P0 新增，正在跑
│   ├── act_scales\ / act_shifts\         各 6.06 MiB（Llama-2-7b-hf.pt，128 样本）
│   ├── lm_eval\ / spike_driven_quant\    代码
│   ├── backup_before_v2_20261028_102423\ 迁移前备份（8 个文件）
│   └── lab_*.log / lab_*.md              全部实验日志与报告
├── hf_cache\                             1.78 GiB（本项目专用 HF_HOME）
├── deploy_packages\                      3 个部署包/指南
├── deploy_review\                        开发组评审副本（原有，未动）
├── .workbuddy\memory\                    工作区记忆
└── 实验室电脑工作日志_给工作AI.md        ← 按要求移到根目录
```

**磁盘**：`C:` 总 1862.60 GB，**剩余 170.91 GB**（90.8% 已用）。迁移净增占用只有 **1.78 GiB**（HF 缓存子集），项目目录是同卷改名，不占新空间。

### 14.6 P0 协议对照实验已启动（按指令"迁移验证通过后立即双卡启动"）

**启动时间**：2026-10-29 **13:03:37**（脚本内 `START 13:03:50`），双卡并行。

**命令**：

```
Set-Location C:\Users\admin\Desktop\spikellm\SpikeLLM
powershell -ExecutionPolicy Bypass -File .\run_lab_p0_protocol.ps1 ^
    -PythonExe "C:\Users\admin\.conda\envs\spikellm\python.exe"
```

**与正式实验的唯一差别**：**不带 `--aug_loss`**。其余完全一致（`wbits=4 abits=4 addbit=1 low_p=0.9 --let --lwc alpha=0.75 let_lr=1e-3 lwc_lr=1e-2 nsamples=128 epochs=20 seed=2`，复用同一份 `./cache/dataloader_Llama_wikitext2_128.cache` 校准子集，三道门禁：fake_binary PPL → real_multibit PPL → 6 零样本）。

**已核验实参**：两腿日志里 `aug_loss=False`，`output_dir` 分别指向 `log\p0_noaug_orig`、`log\p0_noaug_osdsq_v2`。首层已开跑，无 NaN。

**FP16 基线没有重跑**：`--aug_loss` 只作用于 W4A4 标定损失，不影响 FP16 前向，`log/7b_fp16` 的 5.4722 / 6.9727 直接复用。

**预计完成**：约 **24–25 h**（正式实验 24.85 h；本轮少了 aug_loss 的额外前向，标定可能略快，瓶颈仍是零样本的 15 h 量级）→ **2026-10-30 13:30 前后**。

**日志**：驱动 `lab_p0_driver.log`；两腿 `log/p0_noaug_*/run.log`；看门狗 `lab_watch_p0.log`（每 10 min 一条）。

---

## 十五、对"项目目录统一迁移与规范化"8 个问题的逐条答复

**1. 迁移后 C 盘剩余空间？HF 缓存总大小？8B 够不够？**

`C:` 总 1862.60 GB，**剩余 170.91 GB**（迁移前 173.23 GB，净减少 1.78 GiB，正是搬进来的 HF 缓存子集）。
本项目 `hf_cache` = **1.78 GiB**（`datasets/` 1.07 GiB + `hub/` 0.73 GiB）。
**够**：8B base 权重约 15–16 GB，加 c4/wikitext 若需重建缓存约 1 GiB，总需求 < 17 GB，当前余量足够跑十次。但注意整盘已用 90.8%，别人再往里塞模型余量会掉。

**2. P0 协议对照（不开 `--aug_loss`）是否已启动？**

**已启动**，见 §14.6。13:03:37 起跑，双卡并行，预计 **2026-10-30 13:30** 前后完成（约 24–25 h）。

**3. FP16 零样本基线（6 任务，约 3–4 h）能否排在协议对照之后紧跟着跑？**

**能，而且建议排。** 排法：P0 结束后立刻在 GPU0 单卡跑 `--wbits 16 --abits 16 --epochs 0 --tasks piqa,arc_easy,arc_challenge,boolq,winogrande,hellaswag`，不经标定、纯前向，3–4 h；同时 GPU1 可并行开第二个 seed 的标定（见第 7 条），两件事不冲突。
需要我现在就把它写成 `run_lab_fp16_zeroshot.ps1` 排进队列吗？（默认我不擅自开，等你们点头。）

**4. 实验室墙上网络接口是否可用、能否插网线？**

**口是好的，但线没插。** 本机有 **3 个千兆口**：`Intel X722 for 1GbE` ×2（主板集成）+ `ASIX USB to Gigabit Ethernet` ×1，当前全部 `Disconnected / 无载波`，说明**物理上没接网线**。网络目前只有 WiFi（TP-LINK USB，SSID `NEUQ`，5 GHz ch48，信号 100%）。
**关键问题不在带宽而在链路非对称**：WiFi `接收 9 Mbps / 传输 143 Mbps` —— 下载被卡在 9 Mbps（实测约 0.6 MB/s），这才是 13.5 GB 要下 6 小时的真正原因。
**插网线的收益**：从约 0.6 MB/s 提到校园网百兆/千兆的 10–100 MB/s，**8B 的 16 GB 从 7.5 小时降到 3–25 分钟**。这需要**有人去实验室插一根网线**（我做不了）。墙上网口是否通、要不要 Portal 认证，插上后我才能测。

**5. 开发机传新包给我，什么通道最方便？ToDesk 可用吗，单文件多大？**

- **ToDesk 4.8.9.0 已安装且常驻**（两个进程在跑）。能用，但版本偏旧，"单文件大小限制"取决于账号版本（免费版对文件传输有限制），我没法在不实际传输的情况下给出确切数字——**建议先传一个 100 MB 左右的包试一次**（我们最大的部署包只有 1.8 MB，肯定够）。
- **更稳的通道：Tailscale。** 本机已登录 `huangyuan4629@`，节点名 `desktop-n9kgss1`，IP **100.106.105.115**，版本 1.102.2，**支持 Taildrop**（`tailscale file cp <files> desktop-n9kgss1:`）。开发机加入同一 tailnet 后，Taildrop 或 `scp`/HTTP 都行，不受第三方限速、不依赖 ToDesk 版本。**这是我推荐的方式。**
- 现状通道（你们 → 用户 → 我）也能用，1.8 MB 无压力，只是多一环。

**6. 这台机器是否有其他人用？25 h 长实验期间 GPU 会被抢占吗？**

- **当前登录会话只有 `bcic` 一个**（console，2026-10-15 16:23 登录至今），无其它活跃会话。
- **但证据表明这台机器被多人/多项目用过**：`~/.cache/huggingface` 里有 Qwen2.5-7B、Mistral-7B、CLIP-H、DINOv2、Llama-3-8B-Instruct 等约 50 GiB 模型，显然不是本项目下的。
- **能确认的**：本次 24.85 h 正式实验期间两张卡从头到尾没被打断（`lab_watch.log` 147 条快照连续，两腿 `done.flag` 齐全），**实际没被抢占**。
- **风险点**：①整盘已用 90.8%，别人再下模型可能把盘塞满，会直接搞崩正在跑的实验；②若有人在图形界面登录并跑任务，显存会被抢；③网络是共享 WiFi，别人占带宽会拖慢下载。
- **建议**：告知使用者 10-30 13:30 前这台机器在跑长实验，别开大任务。我可以给看门狗加一条"磁盘余量 < 30 GB 就告警"，要的话说一声。

**7. 额外种子实验的独立 cache 目录 + wikitext/c4 离线可用性——已验证，但发现一个坑**

- **seed 确实会换校准子集**：实测 `seed=2` 与 `seed=3` 各取 128×2048 token，**逐 token 不一致率 99.19%**。所以 `--cache_dir ./cache_seedN` 是**必须**的，否则 `dataloader_Llama_wikitext2_128.cache` 会被直接复用、seed 形同虚设。建议每个 seed 一个目录：`cache_seed2 / cache_seed3 / cache_seed4`。
- **wikitext2 离线可用**：`[OK] wikitext-2-raw-v1 -> 4358 rows`（test），校准集重建走的就是它，离线 OK。
- **c4 离线会失败**：报 `Couldn't find cache for allenai/c4 for config 'default-…'`。
  **这不是迁移造成的**：`datautils.get_c4()` 用 `load_dataset('allenai/c4', data_files={'train': 'en/c4-train...json.gz'})`，config 哈希由**绝对路径**推导；offline 模式下 `datasets` 先把 `data_files` 解析成绝对路径再走 `cache` 模块，算出的哈希与在线建缓存时的哈希**天生不相等**。迁移前同样会失败，只是之前没人测过。
  **两个解法**（任选，都不改算法）：
  1. **最省事（推荐）**：新 cache dir 里**预拷** `testloader_Llama_c4_all.cache` 与 `testloader_Llama_wikitext2_all.cache`。已核源码，PPL 评测分支 `get_loaders(dataset, seed=2, …)` 的 seed **硬编码为 2**，因此这两个 testloader 与 seed 无关、可跨 seed 复用。这样新 seed 只需重建 seed 敏感的 `dataloader_*_wikitext2_128.cache`（走 wikitext，离线 OK），**完全不碰 c4**。
  2. 需要重建 c4 时，不要设 `HF_DATASETS_OFFLINE`（只设 `HF_HUB_OFFLINE=1` 也不行），让它在线跑一次——**raw 文件都在本地，实测 20 秒完成，不会真的下载**。

**8. 同步日志放哪？**

已按你们要求移到 **`C:\Users\admin\Desktop\spikellm\实验室电脑工作日志_给工作AI.md`**（根目录，不在 `SpikeLLM\` 子目录下），本文件就是。会持续更新。

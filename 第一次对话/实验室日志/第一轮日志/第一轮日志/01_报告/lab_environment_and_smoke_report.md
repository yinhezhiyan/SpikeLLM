# 实验室双 3090 环境检查与 Smoke 验证报告

> **⚠️ 路径变更说明（2026-10-29 12:42 追加）**
> 本文正文中的所有 `C:\spikellm\SpikeLLM` 路径已失效。整个项目已迁移到
> `C:\Users\admin\Desktop\spikellm\SpikeLLM`（详见同步日志 §十四）。
> 文中记录的**数值、耗时、失败模式全部仍然有效**，只是路径前缀需要替换。
> 其余内容保持原样，作为 2026-10-27 那个时间点的历史记录。

- 机器：实验室 Windows 工作站（2× RTX 3090）
- 项目根目录：`C:\spikellm\SpikeLLM`（**现已迁至 `C:\Users\admin\Desktop\spikellm\SpikeLLM`**）
- 报告生成时间：2026-10-28 00:30（Asia/Shanghai）
- 执行范围：`run_lab_dual3090_7b_8b.ps1` 的 `check` → `download_7b` → `smoke_7b`（含全部 resume 门禁）
- **未执行**：`7b_parallel`、`download_8b`、`smoke_8b`、`8b_parallel`（按指令在 smoke 后停止）

---

## 0. 结论摘要

| 项目 | 结论 |
|---|---|
| 硬件 | 可用。2× RTX 3090 24 GB，driver 610.88，255.66 GB RAM，C 盘余 109.6 GB，双卡全程无他人抢占 |
| 软件 | 可用，但需要 **8 处代码补丁 + 3 项环境/数据修正**（见 §4）。Python 3.11.11 + torch 2.6.0+cu126 + transformers 4.44.2 |
| 网络 | 可用但极慢（约 0.6 MB/s），根因已定位（§2）。`huggingface.co` 不可达，走 `hf-mirror.com` |
| 模型数据 | 完整。权重 13,476,876,272 B，无残留 `.incomplete`；WikiText-2 / C4 缓存齐备；6 个零样本数据集全部就绪 |
| 标定完成度 | **两条腿 32/32 层全部完成** |
| checkpoint | 两腿均生成 `omni_parameters.pth`（8,983,506 B） |
| 显存峰值 | 标定 12,618 MB（Orig）/ 12,586 MB（Ours）；推理 22,553 MB（PIQA 阶段） |
| OOM / 崩溃 | **无** |
| NaN | **有**，梯度范数 NaN：Orig 层 1/24/30/31，Ours 层 1/30/31（§7 P1） |
| 是否可称"全链路评测通过" | **不可以**。见 §8 覆盖边界 |

### 三个数字（最重要的结论）

| 配置 | WikiText-2 | C4 | 相对 FP16 |
|---|---|---|---|
| **FP16 基线（本机实测）** | **5.4722** | **6.9727** | 1.00× |
| Original SpikeLLM W4A4（8 样本×1 epoch） | 59.17 | 64.20 | 10.8× / 9.2× |
| OSDSQ v2 W4A4（8 样本×1 epoch） | 26.72 | 34.46 | 4.9× / 4.9× |

FP16 基线与 Llama-2-7B 公认值（WikiText-2 ≈ 5.47、C4 ≈ 6.97）**逐位吻合**，说明数据、tokenizer、2048 窗口 NLL 统计、权重拼接全部正确。

零样本侧同样给出一致证据：

| 配置 | PIQA acc | acc_norm |
|---|---|---|
| Original SpikeLLM W4A4 | 0.5691 | 0.5604 |
| OSDSQ v2 W4A4 | **0.6045** | **0.5952** |

（随机基线 0.50，Llama-2-7B FP16 约 0.79。两腿都远未收敛，但 OSDSQ v2 领先 3.5 个点，**方向与 PPL 一致**。）

**因此**：W4A4 的高 PPL 不是评测 bug，而是 8 样本 × 1 epoch 远不足以标定。OSDSQ v2 在同一预算下把恶化从约 10× 压到约 4.9×，方向明确、幅度显著，但**绝对不能当作论文数字**。

---

## 1. 硬件与软件环境

### 1.1 硬件

| 项 | 值 |
|---|---|
| 操作系统 | Microsoft Windows 11 专业版，Build 22631，Version 10.0.22631 |
| CPU | Intel Xeon Gold 6133 @ 2.50 GHz，20 核 / 40 线程 |
| 内存 | 255.66 GB |
| 磁盘 | **仅 C: 盘**（本机无 D: 盘），总 1862.6 GB，可用 109.61 GB |
| GPU | 2× NVIDIA GeForce RTX 3090，各 24576 MiB，compute capability 8.6 |
| 驱动 | 610.88 |
| 执行前占用 | 双卡空闲，仅桌面 C+G 进程 |

> 操作指南假定项目在 `D:\spikellm\SpikeLLM`；本机无 D 盘，实际部署在 `C:\spikellm\SpikeLLM`。脚本内部路径均为相对路径，不受影响。

### 1.2 Python 与依赖

```
C:\Users\admin\.conda\envs\spikellm\python.exe
```

| 包 | 版本 |
|---|---|
| Python | 3.11.11 |
| torch | 2.6.0+cu126（`torch.cuda.is_available() == True`） |
| CUDA runtime | 12.6 |
| transformers | **4.44.2** |
| datasets | 3.5.0 |
| accelerate | 1.7.0 |
| tokenizers | 0.19.1 |
| huggingface-hub | 0.30.2 |
| numpy | 1.26.4 |
| safetensors | 0.5.3 |
| sentencepiece | 0.2.0 |
| auto_gptq | **未安装**（仅 `--real_quant` 需要） |

> ⚠️ PATH 上的默认 `python` 是 WorkBuddy 自带的 3.13.x，**没有 torch**。所有调用必须带
> `-PythonExe "C:\Users\admin\.conda\envs\spikellm\python.exe"`，否则 `check` 直接判失败。

---

## 2. 网络环境（含"为什么慢"的定位）

### 2.1 可达性

| 目标 | 结果 |
|---|---|
| `huggingface.co` | **不可达**（连接超时） |
| `hf-mirror.com` | 可达，RTT 约 204 ms，丢包约 25% |
| 本地代理（xray 10808 / Clash 7890） | 端口在听，但对 HF 请求返回 0 字节，**不可用** |

### 2.2 为什么慢（已定位，不是脚本问题）

实测有效带宽 **约 0.6 MB/s（609–682 KB/s）**；13.47 GB ÷ 0.6 MB/s ≈ 6.2 小时，与实测（约 6 小时，含 1.5 小时中断）吻合。说明不是卡在坏连接上，而是链路本身只有这个速度。

按贡献度排序：

1. **WiFi 上下行严重不对称**：`netsh wlan show interfaces` 显示 RX link rate 被钉在 **9 Mbps**，TX 为 143 Mbps。信号 96–100%、5 GHz ch48，不是弱信号，而是校园网 `NEUQ` 对该客户端的下行做了限制/协商降级。
2. **校园网出口 + 镜像侧限速叠加**：公网出口为中国联通秦皇岛（IPv6）。清华 TUNA 实测每连接约 7 KB/s（4 并发合计 29 KB/s）。
3. **只有一条 WiFi 链路**：3 个千兆网口全部未连接，唯一在用的 TP-LINK USB 无线网卡承担全部流量。
4. **ToDesk 常驻会话**：远控画面与下载共用同一条 9 Mbps 链路。

**影响**：只影响时间，不影响正确性。但 8B（约 16 GB）+ C4 缓存预计还要 **5 小时以上**，开跑前请先决定是否值得等，或改有线。

---

## 3. 实际执行的命令、耗时与退出码

| # | 步骤 | 时间窗 | 耗时 | 退出码 | 日志 |
|---|---|---|---|---|---|
| 1 | `-Mode check` | 15:26，01:1x 复跑 | <1 min | **0（PREFLIGHT CHECK PASSED）** | `lab_check_final.log` |
| 2 | `-Mode download_7b`（第 1 次） | 15:2x→17:04 | 1 h 37 min | 0（**静默中断**） | `lab_download_7b.log` |
| 3 | `-Mode download_7b`（续传） | 20:58→21:10 | 12 min | 1（C4 缓存构建崩溃） | `lab_download_7b_retry1.log`、`_c4fix.log` |
| 4 | `-Mode download_7b`（修好后） | 21:10→21:14 | 4 min | **0（成功）** | `lab_download_7b_c4fix2.log` |
| 5 | `-Mode smoke_7b`（第 1 次） | 21:20 | 秒退 | 1（PowerShell stderr 陷阱） | `lab_smoke_7b.log` |
| 6 | 同上（第 2 次） | 21:22 | 秒退 | 1（`--net` 被 choices 拒绝） | `lab_smoke_7b_run2.log` |
| 7 | 同上（第 3 次） | 21:23 | 秒退 | 1（rotary_emb 设备不一致） | `lab_smoke_7b_run3.log` |
| 8 | 同上（第 4 次） | 21:30 | 约 1 min | 1（attention mask 多 1 列） | `lab_smoke_7b_run4.log` |
| 9 | 同上（第 5 次） | 21:30→22:47 | 77 min | 两条腿标定 + PPL 成功；**PIQA 门禁因网络失败** | `lab_smoke_7b_run5.log`（152 KB） |
| 10 | 补跑 3 个门禁 | 23:33→01:05 | 1 h 32 min | **0（全部通过）** | `lab_gate_orig_piqa.log`、`lab_gate_ours_piqa.log`、`lab_gate_ours_real_capture.log` |
| 11 | FP16 基线（诊断） | 23:36→23:44 | 8 min | **0** | `lab_fp16_baseline_ref.log` |
| 12 | 零样本数据集修复 | 23:40→23:50 | 10 min | **0** | `lab_dataset_repair2.log` |

每次失败都保留了完整日志，失败原因全部定位到具体根因，没有盲目重试或换源绕过。

---

## 4. 改动清单（8 处代码 + 3 项环境/数据）

> 原则：**没有修改量化算法本身**。代码改动全部是为适配 `transformers 4.44.x` / `datasets 3.5` / PowerShell 5.1 / 本机 symlink 缺陷，逐条给出等价性论证。代码里都留了 `NOTE (lab machine fix 2026-10-27)`，可直接 `grep -rn "lab machine fix"`。

### 4.1 代码（8 处）

| # | 文件 | 改动 | 原因与安全性 |
|---|---|---|---|
| 1 | `run_lab_dual3090_7b_8b.ps1` | `$ErrorActionPreference`：`"Stop"` → `"Continue"` | PowerShell 5.1 下 `2>&1 \| Tee-Object` 把原生命令 stderr 当 terminating error；`main.py` 一启动就打 `RequestsDependencyWarning`，脚本当场死（日志为空、秒退）。门禁语义不变，因为后面仍检查 `$LASTEXITCODE` |
| 2 | `main.py` | `--net` 去掉 `choices=net_choices` | `net_choices` 不含 `Llama-2-7b-hf`/`Meta-Llama-3-8B` 这类目录名。`args.net` 只用于家族子串判断与 `act_scales/{net}.pt` 拼路径，且下一行 `assert args.net in net_choices` 本就注释掉 |
| 3 | `spike_driven_quant/spike_omniquant.py` | 新增 `model.model.rotary_emb.to(dev)` 与对称 `.cpu()` | `transformers>=4.42` 把 rotary 提到 `LlamaModel` 层，原代码只搬 `embed_tokens`/`norm`，`inv_freq` 留 CPU → capture 前向 `RuntimeError: cpu and cuda:0`。纯设备搬运 |
| 4 | `models/spike_llama_layer.py` | rotary 改 4.44 签名：`cos, sin = self.rotary_emb(x, position_ids)`，再 `apply_rotary_pos_emb(q,k,cos,sin)` | 4.44 返回 3D cos/sin 且内部自行 unsqueeze。**RoPE 数学完全未变** |
| 5 | `models/spike_llama_layer.py` | eager mask 多 1 列时切掉尾列 | 4.44.x eager 分支 `target_length = past_seen + seq + 1`（`modeling_llama.py` 约 1082 行），**上游 bug**，已用 tiny model 最小复现。多出的列被 `triu(diagonal=1)` 全 mask，切掉对数值零影响 |
| 6 | `run_lab_dual3090_7b_8b.ps1` | 顶部新增 `HF_ENDPOINT=https://hf-mirror.com`、`HF_DATASETS_TRUST_REMOTE_CODE=1`（未设置时才设） | **§7 P0-B**：`HF_ENDPOINT` 原只在 `lab_preflight_and_download.py` 里设，`main.py`/`lm_eval` 是另一进程拿不到，零样本一联网就撞超时 |
| 7 | **新增 `lab_compat.py`**，并在 `main.py` / `lab_preflight_and_download.py` / `generate_act_scale_shift.py` 最顶部 import | 把 `huggingface_hub.file_download.are_symlinks_supported` 打回 `False`，逼 hub 走复制分支 | **§7 P0-A**，本机最深的坑。对正常机器无害（最多多占磁盘） |
| 8 | `run_lab_dual3090_7b_8b.ps1` | STEP 1 FP16 基线加 `--epochs 0` | **§7 P0-C**，否则正式实验第一步 `AssertionError` 直接终止 |

另外 `QuantLlamaDecoderLayer.forward` 原本用 `*args/**kwargs` 吃新参数，但 4.44 传的 `cache_position` / `position_embeddings` 需显式接收，已补签名（不改变计算路径）。

### 4.2 环境/数据（3 项）

| 项 | 内容 | 原因 |
|---|---|---|
| transformers | 锁定 **4.44.2** | 4.42.4 的 rotary 在 `LlamaAttention` 内（无 `org_module.rotary_emb`），更不兼容；4.44.x 只需补丁 4、5 即可跑通 |
| C4 缓存 | 早期手工把 blob 复制进 snapshot（`README.md`、`en/c4-train...json.gz` 319 MB），并用 curl 从 hf-mirror 直下 `c4-validation.00000-of-00008.json.gz`（40 MB） | 当时是 symlink 缺陷（§7 P0-A）导致；**后续已由 `lab_compat.py` 根治**，不再需要手工操作 |
| 模型目录 | 从旧迁移副本 `C:\DingXL\SpikeLLM_KV_study\SpikeLLM` 复制 `models/` 与 `lm_eval/models/` | 部署 zip 打包时按目录名排除了任何 `models` 目录。已核对 `SpikeLinear.__init__` 签名一致、`lm_eval/models/` 逐字节相同 |

> 数据集修复过程中，把 4 个损坏的数据集缓存目录**移动**（非删除）到 `datasets--*.broken_<ts>`，可原样移回。

---

## 5. 下载完整性核对

### 5.1 权重（`models/Llama-2-7b-hf/`）

| 文件 | 大小 (B) |
|---|---|
| `model-00001-of-00002.safetensors` | 9,976,578,928 |
| `model-00002-of-00002.safetensors` | 3,500,297,344 |
| `model.safetensors.index.json` | 26,788 |
| `config.json` | 583 |
| `tokenizer.json` | 1,842,764 |
| `tokenizer.model` | 499,723 |
| `tokenizer_config.json` | 746 |
| `special_tokens_map.json` | 435 |

合计 **13,476,876,272 B ≈ 12.55 GiB**，`.incomplete` 残留 **0**，无 `.bin` 分片（符合 `ignore_patterns` 设定）。

### 5.2 数据集

| 文件/数据集 | 规模 | 状态 |
|---|---|---|
| `cache/dataloader_Llama_wikitext2_128.cache` | 25,141,987 B | ✅ |
| `cache/dataloader_Llama_wikitext2_8.cache` | 23,131,469 B | ✅（smoke 用） |
| `cache/testloader_Llama_wikitext2_all.cache` | 5,465,186 B | ✅ |
| `cache/testloader_Llama_c4_all.cache` | 4,195,628 B | ✅ |
| `piqa` | 16113 / 3084 / 1838 | ✅ |
| `ai2_arc ARC-Easy` | 2251 / 2376 / 570 | ✅ |
| `ai2_arc ARC-Challenge` | 1119 / 1172 / 299 | ✅ |
| `boolq` | 9427 / 3270 | ✅ |
| `winogrande` | 40398 / 1767 / 1267 | ✅ |
| `hellaswag` | 39905 / 10003 / 10042 | ✅ |

六个零样本数据集**全部可加载**——这是 `7b_parallel` 零样本阶段的必要条件，修复前其中 4 个是坏的。

---

## 6. Smoke 执行结果

### 6.1 命令

```powershell
Set-Location C:\spikellm\SpikeLLM
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 `
  -Mode smoke_7b -PythonExe "C:\Users\admin\.conda\envs\spikellm\python.exe"
```

两条腿的标定参数（脚本硬编码，**不是论文正式设置**）：

```
--wbits 4 --abits 4 --addbit 1 --low_p 0.9 --let --lwc --aug_loss
--alpha 0.75 --let_lr 0.001 --lwc_lr 0.01 --nsamples 8 --epochs 1 --seed 2 --eval_ppl
```

### 6.2 标定完成度

| 腿 | 层数 | 完成 | 耗时 |
|---|---|---|---|
| Original SpikeLLM（`--orig_spikellm`） | 32 | **32/32** | 760.0 s |
| OSDSQ v2（默认） | 32 | **32/32** | 773.1 s |

`=== Start quantize layer 0 ===` … `31` 全部出现，每层都有 `loss/norm/max memory_allocated` 记录，**无跳层、无中断**。

### 6.3 逐层标定 loss

两腿均为"逐层递增"的正常形态；**OSDSQ v2 在每一层都低于 Original**（比值 0.6–0.75）。

| 层 | Original loss | Orig grad-norm | OSDSQ v2 loss | Ours grad-norm |
|---|---|---|---|---|
| 0 | 0.000163 | 1.56e-4 | 0.001476 | 1.41e-3 |
| 1 | 0.0766 | **nan** | 0.0372 | **nan** |
| 2 | 0.0435 | 8.14e-4 | 0.0362 | 1.58e-3 |
| 3 | 0.0464 | 9.92e-4 | 0.0407 | 1.43e-3 |
| 4 | 0.0504 | 5.48e-4 | 0.0438 | 6.45e-4 |
| 5 | 0.0560 | 4.73e-4 | 0.0479 | 5.22e-4 |
| 6 | 0.0658 | 1.82e-3 | 0.0545 | 1.35e-3 |
| 7 | 0.0770 | 1.81e-3 | 0.0630 | 1.18e-3 |
| 8 | 0.0899 | 2.28e-3 | 0.0719 | 1.87e-3 |
| 9 | 0.1022 | 1.45e-3 | 0.0808 | 8.53e-4 |
| 10 | 0.1181 | 1.70e-3 | 0.0915 | 1.25e-3 |
| 11 | 0.1373 | 6.26e-3 | 0.1032 | 2.85e-3 |
| 12 | 0.1556 | 2.39e-3 | 0.1152 | 1.48e-3 |
| 13 | 0.1854 | 2.27e-3 | 0.1327 | 1.32e-3 |
| 14 | 0.2058 | 1.89e-3 | 0.1439 | 9.42e-4 |
| 15 | 0.2532 | 4.24e-3 | 0.1693 | 1.90e-3 |
| 16 | 0.3218 | 2.56e-3 | 0.2147 | 1.57e-3 |
| 17 | 0.3791 | 3.81e-3 | 0.2479 | 1.50e-3 |
| 18 | 0.4719 | 8.19e-3 | 0.3037 | 3.80e-3 |
| 19 | 0.5690 | 3.93e-3 | 0.3667 | 2.23e-3 |
| 20 | 0.7222 | 4.50e-3 | 0.4634 | 2.70e-3 |
| 21 | 0.8553 | 2.47e-3 | 0.5483 | 1.28e-3 |
| 22 | 1.0596 | 5.34e-3 | 0.6772 | 5.12e-3 |
| 23 | 1.2220 | 3.04e-3 | 0.7837 | 2.34e-3 |
| 24 | 1.4383 | **nan** | 0.9255 | 5.43e-3 |
| 25 | 1.6232 | 6.01e-3 | 1.0412 | 3.42e-3 |
| 26 | 1.8995 | 7.02e-3 | 1.2208 | 4.55e-3 |
| 27 | 2.1509 | 5.03e-3 | 1.3790 | 3.49e-3 |
| 28 | 2.4905 | 8.01e-3 | 1.5984 | 4.02e-3 |
| 29 | 2.9007 | 1.21e-2 | 1.8659 | 6.11e-3 |
| 30 | 4.8225 | **nan** | 3.0752 | **nan** |
| 31 | 16.3736 | **nan** | 10.1071 | **nan** |

### 6.4 PPL：三种配置对照（含本机实测 FP16 基线）

| 配置 | WikiText-2 | C4 | 相对 FP16（WikiText-2 / C4） |
|---|---|---|---|
| **FP16**（`--wbits 16 --abits 16`） | **5.4722** | **6.9727** | 1.00× / 1.00× |
| Original・`fake_binary_simulate` | 59.17 | 64.20 | 10.81× / 9.21× |
| OSDSQ v2・`fake_binary_simulate` | **26.72** | **34.46** | **4.88× / 4.94×** |
| Original・`real_multibit_simulate` | 66.20 | 68.93 | 12.10× / 9.89× |
| OSDSQ v2・`real_multibit_simulate` | **30.89** | **39.89** | **5.65× / 5.72×** |

- OSDSQ v2 vs Original：`fake_binary` −54.8% / −46.3%；`real_multibit` −53.3% / −42.1%
- `real_multibit_simulate` 比 `fake_binary` 更差（Orig 59.17→66.20），符合"更贴近真实多位脉冲"的预期，**没有出现异常反转**
- FP16 与文献值逐位吻合，**证明评测链路准确**

### 6.5 checkpoint 与显存

| 腿 | checkpoint | 峰值显存（标定） |
|---|---|---|
| Original | `log/7b_smoke_orig/omni_parameters.pth`，8,983,506 B | 12,618 MB |
| OSDSQ v2 | `log/7b_smoke_osdsq_v2/omni_parameters.pth`，8,983,506 B | 12,586 MB |

PIQA 阶段单卡占用 **22,553 MiB**（含 KV/评测开销），仍在 24 GB 内但余量仅约 2 GB。

### 6.6 异常情况

| 项 | 结果 |
|---|---|
| OOM | **无** |
| Python 异常 | 标定/PPL 阶段 **无**；PIQA 门禁首次因网络失败（§7 P0-B），修复后通过 |
| NaN | 梯度范数 NaN（§7 P1）；**未出现** `Loss is NAN, stopping training`（loss 本身有限） |
| 显存泄漏迹象 | 无，逐层 `max memory_allocated` 单调缓增后稳定在 12.6 GB |

### 6.7 四个 resume 门禁（checkpoint resume + real_multibit + PIQA）

> **说明（透明度）**：`-Mode smoke_7b` 首次执行到第 2 个门禁（Original PIQA）时因 §7 P0-B 的网络问题抛异常退出，脚本自身的 4 个门禁没有跑完。
> 修好环境变量后，我用**与脚本完全相同的参数**逐条补跑（`lab_gate_*.log`），4 条全部退出码 0。数值如下。

| 门禁 | 结果 | 数值 |
|---|---|---|
| Original：resume + `real_multibit_simulate` + `--eval_ppl` | ✅ 通过 | WikiText-2 66.1974 / C4 68.9262 |
| Original：resume + PIQA | ✅ 通过 | **acc 0.5691**（stderr 0.0116）/ acc_norm 0.5604 |
| OSDSQ v2：resume + `real_multibit_simulate` + `--eval_ppl` | ✅ 通过 | WikiText-2 30.8939 / C4 39.8881 |
| OSDSQ v2：resume + PIQA | ✅ 通过 | **acc 0.6045**（stderr 0.0114）/ acc_norm 0.5952 |

> PIQA 随机基线为 50%。两腿都高于随机，但都明显低于 FP16（Llama-2-7B PIQA 约 0.79），与"8 样本×1 epoch 未收敛标定"一致。
> **OSDSQ v2 在 PIQA 上也优于 Original：acc 0.6045 vs 0.5691（+3.5 个点），acc_norm 0.5952 vs 0.5604（+3.5 个点）**，方向与 PPL 一致，是一条互相印证的证据。
> 注意 stderr 约 0.0115，差值 0.035 ≈ 3σ，统计上可辨但不算强；**正式实验须用 128 样本 × 20 epoch 重测**后再下结论。

---

## 7. 必须上报的问题（按严重度）

### P0-A — 本机 `os.symlink()` 静默失效，会摧毁 HF 缓存（已根治）

最小复现：

```python
os.symlink(rel_src, dst)   # 不抛异常
os.path.lexists(dst)       # >>> False   ← 链接根本没建出来
```

同一目录下 `shutil.copyfile` **完全正常**。

`huggingface_hub` 用 `are_symlinks_supported()` 决定要不要软链，而该探测**只检查 `os.symlink()` 是否抛异常，不校验结果**。于是本机被误判为支持软链：hub 保留 `blobs/<etag>` 后去 `snapshots/<rev>/` 建指针，指针静默失败，快照目录空着。

**已造成的实际故障**：

| 表现 | 触发点 |
|---|---|
| `HFValidationError: Repo id must use alphanumeric chars ... '<abs path>/README.md'` | C4 缓存构建崩溃；boolq / winogrande / hellaswag |
| `FileNotFoundError: ...\snapshots\<rev>\ARC-Challenge\train-00000-of-00001.parquet` | arc_challenge |
| blob 已下好（6570 B）但 `snapshots/<rev>/` 为空 | boolq 等 |

**修复**：`lab_compat.py` 把 `are_symlinks_supported` 打回 `False`，逼 hub 走 `shutil.move/copyfile`。**验证：6 个零样本数据集全部加载成功**（见 §5.2）。

> 这个 shim 建议回灌主仓，对正常机器无害。

### P0-B — `HF_ENDPOINT` 没传给 `main.py` / `lm_eval`（已修）

`HF_ENDPOINT` 只在 `lab_preflight_and_download.py` 里设置，`main.py` 是**另一个进程**。PIQA 门禁首次直接死在：

```
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='huggingface.co', port=443):
Read timed out. (read timeout=10)
```

另外 `datasets 3.5` 加载 `piqa`（自定义代码数据集）**必须** `trust_remote_code=True`，而 `lm_eval/base.py` 的 `load_dataset(...)` 没传。用 `HF_DATASETS_TRUST_REMOTE_CODE=1` 环境变量绕过，**不必改 lm_eval**。

### P0-C — `7b_parallel` 的 STEP 1 会直接 `AssertionError`（已修，**必须回灌**）

STEP 1 的 FP16 基线只传 `--wbits 16 --abits 16 --eval_ppl`，没传 `--let/--lwc`，而 `--epochs` 默认 **20**。`main.py` 第 262 行是**无条件**的：

```python
if args.epochs > 0:
    assert args.lwc or args.let      # wbits=16 也照样触发
```

**实测已复现**：`AssertionError`，退出码 1。因后面有 `if ($LASTEXITCODE -ne 0) { throw }`，**整个 `7b_parallel` / `8b_parallel` 会在第 1 步终止，连标定都进不去**。

修法：给 STEP 1 加 `--epochs 0`（`wbits=16` 时 `if args.wbits < 16 or args.abits < 16` 为假，量化整段跳过，epochs 无意义）。已验证可跑通并拿到基线。

### P1 — 梯度范数 NaN

- Original：层 **1, 24, 30, 31**；OSDSQ v2：层 **1, 30, 31**。层 1 两腿皆有 ⇒ 系统性，不是偶发。

机制（已读代码确认）：`utils.py` 的 `NativeScalerWithGradNormCount` 用 `torch.cuda.amp.GradScaler`，先 `scale(loss).backward()` 再 `unscale_` 再 `torch.norm(...)`。**loss 本身有限**（层 1 = 0.0766 / 0.0372），但 unscaled 梯度含 inf/nan，范数即为 nan。

**后果**：`GradScaler.step()` 检测到 inf/nan 会**跳过该步**、`update()` 把 scale 减半。所以——

- 权重**不会被污染**（好消息）
- 但 `epochs=1` 下这些层的 LET/LWC **一次都没更新**，等于没训
- 这很可能直接贡献了偏高的 PPL

**判断**：`epochs=20` 下大概率自愈（首次溢出后 scale 减半，后续 epoch 正常更新），但**必须在正式日志里 grep `norm:nan` 确认**。

**待确认**：原机器上跑 Original 时是否也出现？若没有，说明是 torch 2.6 / CUDA 12.6 的 AMP 行为差异，需要认真对待。

**备选诊断**：`--deactive_amp`（走 fp32 分支）。但该分支下 `spike_omniquant.py:214` 不再 `.float()` mask，若 cache 里 mask 是 fp16 会撞 dtype mismatch，需一并处理。建议先在 smoke 上试。

### P2 — 门禁脚本与 smoke 超参不一致

`run_lab_dual3090_7b_8b.ps1` 的 resume 门禁**没传** `--alpha/--let_lr/--lwc_lr/--aug_loss`，走默认 `alpha=0.5, let_lr=0.005, aug_loss=False`，与 smoke 标定的 `alpha=0.75, let_lr=0.001` 不一致（已在 §6.7 的 args 日志中确证）。`--epochs 0 --resume` 下大概率无害，但属于"门禁没覆盖正式路径"的漏洞，建议补齐。

### P3 — `datasets 3.5.0` + `huggingface_hub 0.30.2` 的 `DatasetCard.load` 不兼容

已由 `lab_compat.py` + 环境变量绕过。**缓存一旦被删就会复现**，所以 shim 必须保留。

### P4 — `auto_gptq` 未安装

启动会打 `auto_gptq is required for real quantization`。smoke 走 `fake_binary_simulate` 不受影响；若后续要用 `--real_quant`，需装匹配 CUDA 12.6 / torch 2.6 的 wheel，**Windows 上通常需自行编译**。

---

## 8. Smoke 到底覆盖了什么（明确边界）

> **重要**：**不得**把这次 smoke 表述为"全链路评测通过"。

| 覆盖 | 说明 |
|---|---|
| ✅ 模型加载（safetensors，`attn_implementation=eager`） | 两腿成功 |
| ✅ 首层输入捕获（`Catcher`）、mask / position_ids 复用 | 两腿成功 |
| ✅ act scales / shifts 复用（128 样本统计量） | 命中并生效 |
| ✅ 逐层 32 层标定（`smooth_and_quant_inplace`） | 两腿 32/32 |
| ✅ checkpoint 落盘 + **resume 回读** | 两腿均生成并被后续门禁成功加载 |
| ✅ `fake_binary_simulate` 与 `real_multibit_simulate` 两条 PPL 路径 | 各 2 次，全部出数 |
| ✅ zero-shot 评测链路（lm_eval + PIQA，3676 个 loglikelihood 请求） | 两腿均跑通并出 acc |
| ✅ FP16 基线 | 实测 5.4722 / 6.9727，与文献吻合 |
| ⚠️ **PPL / acc 的绝对数值** | 8 样本 × 1 epoch，远未收敛，**不能作为论文数字** |
| ⚠️ NaN 是否在正式设置下消失 | **未验证**（epochs=1 无法体现） |
| ❌ 128 样本 × 20 epoch 的收敛行为 | 未跑 |
| ❌ 双卡并行（`7b_parallel` 的 `Start-Job` 调度） | 未跑 |
| ❌ 8B 全流程 | 未跑 |

---

## 9. 结论与建议

### 9.1 状态判断

**环境层面：已就绪。** 缺失项（`models/`、`lm_eval/models/`、C4 缓存、act 统计量、6 个零样本数据集、8 处代码兼容）全部补齐，每项都有根因说明与安全性论证。

**结果层面：管路已通，数值不可用。** 唯一可信的量化结论是"OSDSQ v2 在同等标定预算下把 PPL 恶化从约 10× 降到约 4.9×，PIQA 也领先 3.5 个点"，方向与幅度都站得住。

> 三个 P0 级阻塞项（P0-A symlink、P0-B HF_ENDPOINT、P0-C STEP 1 AssertionError）**已在本地修好并验证**，但**必须回灌主仓**，否则你们那边仍是坏的。

### 9.2 开正式实验前建议做的三件事

1. **在正式日志里 grep `norm:nan`**。若 `epochs=20` 下仍出现，就要上 `--deactive_amp` 诊断（注意 P1 里的 mask dtype 问题）。零额外开销。
2. **补齐门禁超参（P2）**，并让门禁跑在正式配置上——resume 路径跑不通等于正式实验的评测阶段全废。
3. **先跑一个计时探针**（单层 + 128 样本 + 1 epoch），确认时间预估（§9.3），避免白等 3 天。

### 9.3 `7b_parallel` 可行性

- **显存**：单腿标定峰值 12.6 GB / 24 GB，双卡各一腿 **安全**；但注意 **PIQA 阶段单卡会到 22.5 GB**，若两腿评测阶段重叠需留意（脚本内是串行，安全）
- **磁盘**：C 盘余 109.6 GB，整套产出（两腿 checkpoint + act 统计 + 日志）1 GB 以内，**安全**
- **时间**：按 smoke 线性外推 12.9 min × (128/8) × 20 ≈ **68 小时/腿**，两腿并行约 68 小时。
  > 这个外推很粗（同层内是 `nsamples/batch_size` 次前向 × epochs，单次前向耗时还与 GPU 状态相关）。**务必先做计时探针**，再决定是否夜间无人值守跑。

### 9.4 8B 预警

8B 权重约 16 GB，按 0.6 MB/s 需 **7.5 小时以上**，加 C4 缓存更久；且本机 HF 缓存只有 `Meta-Llama-3-8B-Instruct`，脚本要的是 base `Meta-Llama-3-8B`。**建议先确认 8B 是否必要**，或走离线拷贝。

---

## 10. 复现命令清单

```powershell
Set-Location C:\spikellm\SpikeLLM
$PY = "C:\Users\admin\.conda\envs\spikellm\python.exe"

# 1) 环境检查（退出码 0 才算通过）
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode check -PythonExe $PY

# 2) 下载 7B + 数据集缓存
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode download_7b -PythonExe $PY

# 3) Smoke（含 4 个 resume 门禁；现已全部通过）
powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode smoke_7b -PythonExe $PY

# 4) FP16 基线（诊断用，正式实验的 STEP 1 会自动跑）
& $PY main.py --model .\models\Llama-2-7b-hf --net Llama-2-7b-hf `
    --wbits 16 --abits 16 --epochs 0 --eval_ppl --output_dir .\log\7b_fp16

# 5) 正式实验（本次未执行，等待确认）
# powershell -ExecutionPolicy Bypass -File .\run_lab_dual3090_7b_8b.ps1 -Mode 7b_parallel -PythonExe $PY
```

关键日志：

```
lab_check_final.log                      环境检查（PREFLIGHT PASSED）
lab_download_7b*.log                     下载（3 次尝试）
lab_smoke_7b_run5.log                    smoke 全量（152 KB）
log/7b_smoke_orig/smoke.log              Original：标定 + fake_binary PPL
log/7b_smoke_osdsq_v2/smoke.log          OSDSQ v2：标定 + fake_binary PPL
lab_gate_ours_real_capture.log           OSDSQ v2：real_multibit PPL（保留下来的副本）
lab_gate_orig_piqa.log                   Original：PIQA
lab_gate_ours_piqa.log                   OSDSQ v2：PIQA
lab_fp16_baseline_ref.log                FP16 基线
lab_dataset_repair2.log                  零样本数据集修复
log/7b_smoke_*/omni_parameters.pth       两腿 checkpoint
```

---

*本报告只覆盖 `check` / `download_7b` / `smoke_7b` 三步及其门禁。正式实验与 8B 全流程均未启动，等待复核确认。*

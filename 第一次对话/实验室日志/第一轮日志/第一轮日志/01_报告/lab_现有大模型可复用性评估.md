# 机器上已有的其他大模型 —— 可复用性评估与确认请求

> 调查时间：2026-10-29 13:35（GMT+8）
> 调查对象：`C:\Users\admin\.cache\huggingface\hub\` 下的全部 13 个模型仓库
> 调查方式：只读。`AutoConfig` / `AutoTokenizer` 实测加载 + `safe_open` 逐个分片 mmap 校验张量数，**未改动、未移动、未删除任何文件**
> 目的：判断能否省掉 8B（16 GB）的下载，以及能否顺手扩几个模型支撑论文

---

## 一、结论（先给答案）

1. **能省掉 8B 的下载，但省不掉的是"base 版"这个要求。** 本机已有 **`Meta-Llama-3-8B-Instruct`（14.97 GiB，完整可用）**，但它是 **Instruct 版**，与开发组要求的 base 版不是同一套权重。要不要用它，取决于论文里 8B 那一行到底想说明什么（见 §七 问题 1）。
2. **Mistral-7B-Instruct-v0.3 和 Llama-3-8B-Instruct 在结构上是 Llama 分支的"零改动可插"级别**（291 个张量、无 attention bias、v0.3 已去掉滑动窗口），只要有 act stats 就能跑。
3. **Qwen2.5-7B-Instruct 不行**，它的 `q/k/v_proj` 带 bias（339 个张量），现有 `QuantLlamaDecoderLayer` 假设的是无 bias 的 Linear，硬上会丢 bias 或直接报错。
4. **另外发现一个可能直接解掉 8B 阻塞的事实**：缓存文件名的 `model_family` 是 `args.net.split('-')[0]`，所以目录名叫 `Meta-Llama-3-8B` 时 family = `"Meta"`，与 7B 的 `"Llama"` **根本不冲突**；只有把目录名写成 `Llama-3-8B` 才会撞。也就是说"7B/8B 缓存键冲突"是**命名造成的，可以规避**（见 §六）。

---

## 二、全盘盘点（13 个仓库，53.95 GiB）

| 仓库 | 体积 | 类型 | 架构 | 层数 | 隐藏维 | KV 头 | 词表 | dtype | 与本项目 |
|---|---|---|---|---|---|---|---|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | 14.97 GiB | LLM | LlamaForCausalLM | 32 | 4096 | 8 | 128256 | bf16 | **候选** |
| `Qwen/Qwen2.5-7B-Instruct` | 14.20 GiB | LLM | Qwen2ForCausalLM | 28 | 3584 | 4 | 152064 | bf16 | 需改代码 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 13.51 GiB | LLM | MistralForCausalLM | 32 | 4096 | 8 | 32768 | bf16 | **候选** |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 7.35 GiB | 视觉 | — | — | — | — | — | — | 无关 |
| `facebook/dinov2-large` | 1.13 GiB | 视觉 | Dinov2Model | 24 | 1024 | — | — | fp32 | 无关 |
| `openai/clip-vit-base-patch32` | 1.13 GiB | 视觉 | CLIPModel | — | — | — | — | — | 无关 |
| `textattack/bert-base-uncased-MNLI` | 0.82 GiB | 分类 | — | — | — | — | — | — | 无关 |
| `timm/swin_base_patch4_window7_224` | 0.33 GiB | 视觉 | — | — | — | — | — | — | 无关 |
| `facebook/opt-125m` | 0.23 GiB | LLM | OPTForCausalLM | 12 | 768 | — | 50272 | fp16 | 玩具，可做冒烟 |
| `timm/swin_small_patch4_window7_224` | 0.19 GiB | 视觉 | — | — | — | — | — | — | 无关 |
| `timm/swin_tiny_patch4_window7_224` | 0.11 GiB | 视觉 | — | — | — | — | — | — | 无关 |
| `bert-base-uncased` | 0.00 GiB | 只有 config/tokenizer，**无权重** | — | — | — | — | — | — | 无关 |
| `NousResearch/Llama-2-7b-hf` | 0.00 GiB | **空占位**，无快照 | — | — | — | — | — | — | 无用 |

合计 **53.95 GiB**。真正与本项目相关的是前三个 LLM，共 **42.68 GiB**。

---

## 三、完整性验证（这三条是硬指标，不是"看起来有"）

三个候选模型全部通过，说明**缓存是真实可用的**（不是上次那种"blobs 有、snapshots 空"的软链损坏状态）：

| 模型 | AutoConfig | AutoTokenizer | safetensors 实读 | 与 index 声明 |
|---|---|---|---|---|
| Llama-3-8B-Instruct | OK | OK（vocab 128000，eos `<|eot_id|>`） | **291 张量 / 4 分片** | 291，**一致** |
| Qwen2.5-7B-Instruct | OK | OK（vocab 151643，eos `<|im_end|>`） | **339 张量 / 4 分片** | 339，**一致** |
| Mistral-7B-Instruct-v0.3 | OK | OK（vocab 32768，eos `</s>`） | **291 张量 / 3 分片** | 291，**一致** |

> 注：这三个仓库的 `blobs/` 目录是空的，权重以**真实文件**形式直接放在 `snapshots/<rev>/` 下。这正是本机 `os.symlink()` 失效后的正常落盘形态（见 `lab_compat.py`），不是损坏。

---

## 四、逐个可用性评估

### 4.1 `Meta-Llama-3-8B-Instruct` —— 结构上零改动可插，但**不是 base 版**

| 检查项 | 结果 | 判断 |
|---|---|---|
| `model_type` | `llama` | ✅ 直接命中 `spike_omniquant.py` 的 `is_llama` 分支 |
| 层布局 | `model.model.layers` + `embed_tokens` + `norm` | ✅ 与 Llama-2 完全一致 |
| 注意力张量 | 只有 `q/k/v/o_proj.weight`，**无 bias** | ✅ 无 bias 假设成立 |
| MLP 张量 | `gate/up/down_proj.weight` | ✅ 与 `pairs={"q_proj":"qkv","o_proj":"out","up_proj":"fc1"}` 兼容 |
| 归一化 | RMSNorm，`rms_norm_eps=1e-5` | ✅ 与 Llama-2 相同 |
| `sliding_window` | 字段不存在 | ✅ 全因果注意力 |
| RoPE | `rope_theta=500000`（Llama-2 是 10000） | ✅ 由 config 驱动，代码不硬编码 |
| GQA | 32 头 / 8 KV 头 | ⚠️ 1.1B 的 GQA-LET 脚本跑过，但 7B 主流程是 MHA，**需开发组确认 LET 对 GQA 的处理** |
| 词表 | 128256（Llama-2 是 32000） | ⚠️ `lm_eval` 任务不受影响，但 `get_wikitext2` 的 token 序列长度会变，缓存需重建 |
| 位置编码上限 | 8192（比我们的 2048 窗口宽松） | ✅ |
| **与 base 版的差异** | Instruct 经过 SFT + RLHF，权重完全不同 | ❌ **不能当作 base 8B 的替代** |

**唯一需要额外做的事**：`act_scales/Meta-Llama-3-8B-Instruct.pt` 与 `act_shifts/...` 要重新生成（`generate_act_scale_shift.py --num_samples 128`，预计 5 min 左右）。

### 4.2 `Mistral-7B-Instruct-v0.3` —— 结构上同样是零改动可插

| 检查项 | 结果 |
|---|---|
| 291 张量 / 3 分片，与 Llama 完全同构 | ✅ 无 attention bias、无 mlp bias |
| `sliding_window = None` | ✅ **v0.3 已经去掉滑动窗口**（v0.1/v0.2 是 4096，那版才需要改 mask 逻辑） |
| `model_type = mistral` | ⚠️ 名字里没有 "llama"，会掉进 `else: raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")` |
| 绕过方式 | `--net` 只是个**字符串**，只用于家族子串判断 + `act_scales/{net}.pt` 拼路径。传一个含 "llama" 的名字（如 `--net Mistral-llama-7B`）就会走 llama 分支，模块布局又能对上 |
| 风险评估 | 这是"靠命名绕过"，不是代码支持。**必须开发组点头**，且必须在 smoke 上验证 |

### 4.3 `Qwen2.5-7B-Instruct` —— 不建议用

| 检查项 | 结果 |
|---|---|
| 339 张量，`q_proj.bias` / `k_proj.bias` / `v_proj.bias` **都存在** | ❌ 现有量化层假设无 bias |
| `model_type = qwen2` | ❌ 没有分支，直接 `ValueError` |
| `sliding_window = 131072` | 在 32k 上下文内等同无滑动窗口，这点不是问题 |
| `num_key_value_heads = 4` | GQA 比率更高 |
| 结论 | 要用得写一个 `QuantQwen2DecoderLayer` 并处理 qkv bias 的量化，**属于算法/实现工作，超出"直接用现成模型"的范围** |

### 4.4 其它

- `facebook/opt-125m`（0.23 GiB，fp16，12 层）：`opt` 分支是现成的，可当**极快的冒烟靶子**（单层秒级），但它不是论文要的模型族。
- 视觉/分类那 6 个（CLIP-H、DINOv2、CLIP-base、BERT-MNLI、3 个 swin）：与 SpikeLLM 的 LLM 量化无关。
- `bert-base-uncased` 只有 config 和 tokenizer，**没有权重**。
- `NousResearch/Llama-2-7b-hf` 是**空占位**（无 snapshot），我们实际用的是项目目录里那份 13 GB 的本地权重。

---

## 五、显存与时间预算（如果用 Llama-3-8B-Instruct）

| 项目 | 7B（已实测） | 8B（估算） | 说明 |
|---|---|---|---|
| 权重（bf16/fp16） | 13.5 GB | 16.1 GB | 词表从 32k 涨到 128k，embed+lm_head 多约 1.0 GB |
| 标定峰值显存 | **18.4 GB**（实测） | **约 21–22 GB**（估） | 24 GB 卡余量只剩 2–3 GB，**有 OOM 风险** |
| 32 层标定耗时 | 6 h 57 min（实测） | 约 7–8 h | 层数与隐藏维相同，主要是词表变大 |
| 6 零样本 | 14.6–16.9 h（实测） | 约 18–22 h | 8B 每 token 更慢 |
| 整轮（两腿并行） | **24.85 h**（实测） | **约 27–31 h** | — |

**建议**：8B 不要直接上 `nsamples=128`，先用 `-Mode smoke_8b`（8 样本 × 1 epoch）确认峰值显存，再决定要不要降 batch 或加 `--deactive_amp` 的替代方案。

---

## 六、⚠️ 一个可能直接解掉 8B 阻塞的发现

开发组之前说"7B/8B 的 WikiText2/C4 缓存都以 `Llama` 为键，并行会互相覆盖"。我核了源码：

```python
# main.py:279
args.model_family = args.net.split('-')[0]

# main.py:100 / 358
cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
```

`model_family` 取的是 `--net` 按 `-` 切分后的**第一段**，实测：

| `--net` 实际值 | `model_family` | 与 7B 的 `"Llama"` 冲突？ |
|---|---|---|
| `Llama-2-7b-hf` | `Llama` | — |
| `Meta-Llama-3-8B` | **`Meta`** | **不冲突** |
| `Meta-Llama-3-8B-Instruct` | **`Meta`** | **不冲突** |
| `Llama-3-8B` | `Llama` | **冲突** |

也就是说：**只要 8B 的模型目录名以 `Meta-` 开头，缓存文件名天然就是 `*_Meta_*`，和 7B 的 `*_Llama_*` 完全分开**。另外脚本本身在 8B 跑完后还会 `Ensure-ActStats` 重新生成统计量，两条腿也不是同时跑 8B。

所以"缓存隔离"这件事可能**根本不用改代码**，只要约定目录命名 + 不要在 7B 与 8B 之间共享 `--cache_dir`。请开发组确认这个判断是否正确 —— 如果对，8B 的排期可以直接提前。

---

## 七、请工作 AI 确认的问题（逐条回答）

1. **8B 那一行到底要说明什么？** 如果论文的 8B 结果是为了"更大模型上同样有效"，那 Instruct 版能不能用？还是必须 base？（我们的判断：Instruct 是另一套权重，替换会改变结论的含义，**默认不用**。）
2. **如果必须 base 版**：现在有网线可插（3 个千兆口都在，只是没接线），插上后 16 GB 大约 3–25 分钟；继续用 WiFi 约 7.5 小时。是等网线，还是先跑别的？
3. **Mistral-7B-Instruct-v0.3 要不要作为第三个模型？** 它结构上和 Llama 完全同构（291 张量、无 bias、v0.3 无滑动窗口），只要 `--net` 传一个含 "llama" 的名字就能走现有分支。**你们允许这种"靠命名绕过家族判断"的做法吗？** 还是希望我加一个显式的 `mistral` 分支（不改算法，只加 `elif`）？
4. **Qwen2.5-7B 是否值得为它写 `QuantQwen2DecoderLayer` 处理 qkv bias？** 我判断这属于实现工作、不在"直接用现成模型"的范围，倾向不做。你们同意吗？
5. **GQA 的 LET 支持到什么程度？** 7B 主流程是 32 头 MHA，而 Llama-3-8B 是 32 头/8 KV 头。你们在 1.1B 上跑过 `gqa_let`，那套逻辑对 8B 的 8 KV 头是否已经验证过？
6. **共享资源的边界**：这 42.68 GiB 的模型不是本项目下载的（`~/.cache/huggingface` 里有 Qwen、Mistral、CLIP、DINOv2 等，明显是多项目共用）。我**没有移动也没有删除**任何文件，只是只读检查。如果要用，是否需要先跟机器的主人打个招呼？我倾向"用可以，但不动它、不复用它的缓存目录"。
7. **如果要用 Llama-3-8B-Instruct，显存峰值预计 21–22 GB（24 GB 卡余量 2–3 GB）**，你们是接受这个风险直接上 128 样本，还是先跑 8 样本 smoke 探底？

---

## 八、我没有做的事（避免误解）

- 没有移动、删除、重命名任何一个模型文件；
- 没有把这三个模型拷进 `Desktop\spikellm\hf_cache`（它们是别人的东西，拷进来会让你结题删除文件夹时误删）；
- 没有修改任何量化代码；
- 没有启动任何实验（P0 仍在跑，两张卡都占着）。

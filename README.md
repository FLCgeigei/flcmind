# FLCMind

中文 | [English](README.en.md)

一个用于**学习**的最小可运行大语言模型项目，完整覆盖 **预训练 → SFT 微调 → 部署推理** 全链路。

- 从零实现的类 Llama 结构（RMSNorm / RoPE / GQA / SwiGLU），并支持 **MoE 混合专家**架构。
- 分词器、训练数据、训练好的权重均已就绪，照着下面的步骤即可一步步跑通。
- 部署提供 **OpenAI 兼容 API**，可直接用 `openai` 客户端调用。

---

## 目录结构

```
flcmind/
├── model/                        # 模型定义与分词器
│   ├── model_flcmind.py          #   FLCMind 模型结构（Config / ForCausalLM）
│   ├── tokenizer.json            #   分词器（词表 6400）
│   └── tokenizer_config.json
├── dataset/                      # 数据加载器与训练数据（数据需从 HF 下载，见下文）
│   ├── lm_dataset.py             #   PretrainDataset / SFTDataset
│   ├── pretrain_t2t_mini.jsonl   #   预训练数据（~1.2GB，解压自 pretrain_t2t_mini.7z）
│   └── sft_t2t_mini.jsonl        #   SFT 对话数据（~1.7GB，解压自 sft_t2t_mini.7z）
├── trainer/                      # 训练脚本
│   ├── train_pretrain.py         #   预训练
│   ├── train_full_sft.py         #   全参数 SFT 微调
│   └── trainer_utils.py          #   公共工具（模型初始化、checkpoint 等）
├── scripts/
│   └── serve_openai_api.py       # OpenAI 兼容 API 服务
├── out/                          # 训练好的模型权重（需从 HF 下载，见下文）
│   ├── full_sft_768.pth          #   Dense SFT 权重（对话能力）★推荐部署
│   └── moe_sft_768_moe.pth       #   MoE SFT 权重（对话能力）
├── checkpoints/                  # 断点续训文件存放处
├── eval_llm.py                   # 命令行推理测试
└── requirements.txt              # 依赖清单
```

---

## 环境准备

```bash
pip install -r requirements.txt
# 另需自行安装与 CUDA 匹配的 torch（requirements 中未固定版本）：
# pip install torch==2.6.0
```

需要一张 CUDA GPU（≥6GB 显存）；仅推理时可用 `--device cpu`，但会慢很多。

---

## 获取数据与权重（Hugging Face）

训练数据与训练好的权重已发布到 Hugging Face，本地仓库不包含这些大文件，首次运行前先下载。

- 数据集：https://huggingface.co/datasets/FLCgeigeigei/flcmind-datasets
- 模型权重：https://huggingface.co/FLCgeigeigei/flcmind

```bash
pip install -U huggingface_hub

# 1) 数据集：下载压缩包 → 解压到 dataset/
hf download FLCgeigeigei/flcmind-datasets --repo-type dataset --local-dir dataset
7z x dataset/pretrain_t2t_mini.7z -odataset
7z x dataset/sft_t2t_mini.7z -odataset

# 2) 模型权重：下载 → 放入 out/
hf download FLCgeigeigei/flcmind --repo-type model --local-dir hf_model
cp hf_model/full_sft_768.pth hf_model/moe_sft_768_moe.pth out/
```

- **只想跳过训练直接对话**：只需执行第 2 步。
- 模型仓库还附带 `model_flcmind.py` 和 `tokenizer*.json`，与项目 `model/` 下的文件相同，无需覆盖。
- 解压需要安装 [7-Zip](https://www.7-zip.org/)；网络受限时可先设置代理，例如 `set HTTPS_PROXY=http://127.0.0.1:7890`（PowerShell 用 `$env:HTTPS_PROXY="http://127.0.0.1:7890"`）。

---

## ⚠️ 重要约定：脚本用相对路径，必须在其所在目录运行

所有脚本都以「自己所在的目录」为工作目录，用 `../` 访问同级的 `model/`、`out/`、`dataset/`。
**因此训练前要先 `cd trainer`，部署前要先 `cd scripts`**，否则会报找不到 `../model` 的错误。
（`eval_llm.py` 在项目根目录运行。）

---

## 快速体验（用下载的权重）

按上文「获取数据与权重」把 SFT 权重下载到 `out/` 后，可跳过训练直接对话：

```bash
cd scripts
python serve_openai_api.py --weight full_sft     # 启动 OpenAI 兼容服务
```

或不起服务，直接命令行对话：

```bash
python eval_llm.py --weight full_sft             # 在项目根目录运行
```

> 想体验 **MoE 版**：`python serve_openai_api.py --use_moe 1 --weight moe_sft`（需先下载 `out/moe_sft_768_moe.pth`，详见下方「MoE 架构」）。

---

## 一、预训练

让模型学会「续写」，具备基础语言能力。

```bash
cd trainer
python train_pretrain.py                          # 用默认数据，输出 ../out/pretrain_768.pth
```

- 数据默认读 `../dataset/pretrain_t2t_mini.jsonl`
- 权重前缀用 `--save_weight` 指定，例如 `--save_weight mymodel` → 生成 `out/mymodel_768.pth`
- 断点续训：`python train_pretrain.py --from_resume 1`

> 预训练权重是**中间产物**，SFT 完成后即可删除（`out/` 只保留最终部署用的 SFT 权重）。

## 二、SFT 微调

在预训练权重基础上继续训练，让模型学会「对话」。

```bash
cd trainer
python train_full_sft.py --from_weight pretrain --save_weight full_sft
```

- 基于预训练权重 `pretrain_768.pth` 继续训练
- 数据默认读 `../dataset/sft_t2t_mini.jsonl`
- 输出 `../out/full_sft_768.pth`

> 已下载 `out/full_sft_768.pth` 时，可跳过此步。

## 三、部署推理

```bash
cd scripts
python serve_openai_api.py --weight full_sft      # 部署对话模型（推荐）
# 无 GPU 时加 --device cpu
```

服务启动在 `http://0.0.0.0:8998`，提供 OpenAI 兼容接口。

### 调用测试

curl：

```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"flcmind-local:latest","messages":[{"role":"user","content":"你好"}],"temperature":0.7}'
```

Python 客户端（流式）：

```python
from openai import OpenAI
client = OpenAI(api_key="sk-any", base_url="http://localhost:8998/v1")
resp = client.chat.completions.create(
    model="flcmind-local:latest",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

### 命令行推理（不起服务）

```bash
python eval_llm.py --weight full_sft
```

---

## MoE 架构（可选）

除默认的稠密（Dense）模型外，本项目还支持 **MoE（Mixture of Experts，混合专家）**：把每层的前馈网络（FFN）替换成多个「专家」，由门控网络（gate）为每个 token 动态挑选 Top-K 个专家参与计算。

**核心思想**：用「稀疏激活」把**参数量**和**计算量**解耦——每个 token 只走 1 个专家（Top-1），所以在推理算力基本不变的前提下大幅扩大模型容量。

| | 稠密 Dense | MoE（4 专家 Top-1） |
|---|---|---|
| 总参数（占显存/体积） | ~65M（`~131MB`） | ~200M（`~400MB`） |
| 激活参数（每 token 实际计算） | ~65M | **~65M**（非 FFN 全激活 + 只算 1 个专家） |
| 单 token 算力 | 1× | ≈1×（几乎相同） |

> 一句话：MoE 拿「更大的显存/存储」换「更大的模型容量」，而单 token 计算量几乎不变。因此它对标的是**同等参数量的大 Dense 模型**（算力省数倍），而不是单个小 Dense。数据越大、推理越频繁，越划算；本项目属小模型 + 小数据，开 MoE 主要用于**学习原理**。

### 启用方式

所有脚本统一用 `--use_moe 1` 开启（默认 `0` 为稠密模型）。开启后权重文件名会**自动带 `_moe` 后缀**（如 `pretrain_768_moe.pth`），用于区分两类权重。

> ⚠️ **训练、SFT、部署、推理必须全程带上一致的 `--use_moe` 值**，否则会因权重形状不匹配而加载失败。

> ✅ 已下载 `out/moe_sft_768_moe.pth`（对话）时，可直接跳到下方第 3 步部署：`python serve_openai_api.py --use_moe 1 --weight moe_sft`。

```bash
# 1. 预训练（MoE）
cd trainer
python train_pretrain.py --use_moe 1 --save_weight pretrain
#   → ../out/pretrain_768_moe.pth

# 2. SFT（MoE）—— --from_weight 传预训练权重的前缀
python train_full_sft.py --use_moe 1 --from_weight pretrain --save_weight full_sft
#   → ../out/full_sft_768_moe.pth

# 3. 部署（MoE）
cd ../scripts
python serve_openai_api.py --use_moe 1 --weight full_sft

# 4. 命令行推理（MoE）
python eval_llm.py --use_moe 1 --weight full_sft
```

> 权重前缀跟着 `--save_weight` 走。例如 `--save_weight moe_model` 会生成 `moe_model_768_moe.pth`，后续步骤的 `--from_weight` / `--weight` 都要用这个前缀 `moe_model`。

**完整示例（前缀自定义，三步前后必须对齐）：**

```bash
# ① 预训练：起个前缀 moe_model
cd trainer
python train_pretrain.py --use_moe 1 --save_weight moe_model
#   → ../out/moe_model_768_moe.pth

# ② SFT：--from_weight 用①的前缀，--save_weight 换个新前缀 moe_sft
python train_full_sft.py --use_moe 1 --from_weight moe_model --save_weight moe_sft
#   → ../out/moe_sft_768_moe.pth

# ③ 部署：--weight 用②的前缀 moe_sft
cd ../scripts
python serve_openai_api.py --use_moe 1 --weight moe_sft
```

> ⚠️ SFT 的 `--save_weight` **别填成预训练的前缀**，否则会覆盖掉预训练权重（同名）。三步的前缀关系是：预训练产出 → SFT 读入并产出新前缀 → 部署读 SFT 的前缀。

### 多卡训练（DDP）

训练脚本支持 `torchrun` 多卡数据并行（部署/推理无需，单卡即可）：

```bash
cd trainer
torchrun --nproc_per_node=2 train_pretrain.py --use_moe 1 --save_weight pretrain
torchrun --nproc_per_node=2 train_full_sft.py --use_moe 1 --from_weight pretrain --save_weight full_sft
```

- `--nproc_per_node` = 使用的卡数；指定用哪几张卡：前面加 `CUDA_VISIBLE_DEVICES=0,1`。
- **`--batch_size` 是「每张卡」的值，不用改**。但有效批量会随卡数放大：
  `有效批量 = batch_size × 卡数 × accumulation_steps`（默认 `32 × 卡数 × 8`）。
  想保持与单卡完全一致的训练行为，把 `--accumulation_steps` 按卡数等比减小即可。
- ⚠️ **Windows 不支持 NCCL**（脚本默认 `backend="nccl"`）。Windows 上多卡易失败，建议单卡训练。

### 相关配置

MoE 超参数在 `model/model_flcmind.py` 的 `FLCMindConfig` 中定义（`use_moe=False` 时全部忽略）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `num_experts` | 4 | 专家总数（越多总参数/容量越大） |
| `num_experts_per_tok` | 1 | 每个 token 激活的专家数（Top-K）；调成 2 则激活参数与算力翻倍 |
| `moe_intermediate_size` | 同 `intermediate_size` | 单个专家 FFN 的中间层维度 |
| `norm_topk_prob` | True | 是否对 Top-K 门控权重归一化 |
| `router_aux_loss_coef` | 5e-4 | 负载均衡辅助损失系数（防止专家旱涝不均） |

> 训练日志会分开打印 `logits_loss`（语言建模损失）和 `aux_loss`（负载均衡损失），二者相加为总损失。

---

## 常见坑

| 现象 | 原因 | 解决 |
|------|------|------|
| `HFValidationError: ... '../model'` | 没在脚本所在目录运行 | 先 `cd trainer` 或 `cd scripts` |
| `CUDA error: device-side assert` | 请求里 `temperature=0.0`，服务端强制采样导致除零 | 客户端把 temperature 设为正数（如 0.7），并**重启服务**清掉污染的 CUDA 上下文 |
| 回复是乱码 / 胡乱续写 | 用了 `pretrain` 预训练权重 | 部署改用 `--weight full_sft` |
| 加载权重形状不匹配 | 启动参数与训练时不一致 | `--hidden_size` / `--num_hidden_layers` / `--use_moe` 必须与训练时相同（默认 768 / 8 / 关） |
| MoE 权重加载失败 / 找不到权重 | 忘了带 `--use_moe 1`，或 `--weight` 前缀写错 | 全程保持 `--use_moe` 一致；`--weight` 传的是**前缀**，脚本自动拼 `_768_moe.pth` |
| `torchrun` 报错 / 指向 `.../elastic/errors.html` | Windows 不支持 NCCL（脚本默认 `backend="nccl"`） | Windows 上改用**单卡** `python ...`；多卡需 Linux 环境 |

---

## 完整流程一览（从零到 API）

```bash
# 1. 预训练
cd trainer
python train_pretrain.py --save_weight mymodel

# 2. SFT
python train_full_sft.py --from_weight mymodel --save_weight mymodel_sft

# 3. 部署
cd ../scripts
python serve_openai_api.py --weight mymodel_sft

# 4. 测试
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"flcmind","messages":[{"role":"user","content":"你好"}],"temperature":0.7}'
```

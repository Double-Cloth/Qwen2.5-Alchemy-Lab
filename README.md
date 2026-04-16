# Qwen2.5-Alchemy-Lab

一个 LoRA 微调实验项目，用来观察“旧蒸馏数据微调新模型”可能带来的负迁移现象。

## 项目目标

- 基座模型：Qwen/Qwen2.5-1.5B-Instruct
- 数据集：yahma/alpaca-cleaned
- 方法：LoRA 微调 + Base/LoRA 对比推理
- 关注点：身份认知、语言一致性、数学推理、代码生成风格

## 目录结构

- src/train_qwen_lora.py：训练脚本（LoRA）
- src/inference.py：交互式推理脚本（支持 Base/LoRA 切换）
- src/compare.py：批量场景对比脚本
- qwen_lora_output/：LoRA 训练输出目录
- samples/：示例图片与结果材料

## 环境准备

推荐 Python 3.10+。

```bash
pip install -r requirements.txt
```

如需指定镜像源，可自行追加 `-i` 参数。

## 快速开始

### 1) 训练 LoRA

```bash
python src/train_qwen_lora.py \
   --model_name Qwen/Qwen2.5-1.5B-Instruct \
   --data_path yahma/alpaca-cleaned \
   --output_dir qwen_lora_output \
   --cache_dir local_cache \
   --epochs 3 \
   --batch_size 2 \
   --grad_accum 16 \
   --lr 2e-4 \
   --max_len 512 \
   --eval_size 1000 \
   --num_proc 1 \
   --num_workers 0
```

说明：

- Windows 下建议 `--num_proc 1 --num_workers 0`，稳定性更高。
- 脚本会自动根据是否有 CUDA 选择精度与设备映射。
- 相对路径参数会按项目根目录解析（不依赖当前终端所在目录）。

### 2) 交互推理

```bash
python src/inference.py \
   --model_name Qwen/Qwen2.5-1.5B-Instruct \
   --lora_path qwen_lora_output \
   --cache_dir local_cache \
   --max_new_tokens 512 \
   --temperature 0.7 \
   --top_p 0.9
```

内置命令：

- `/t` 或 `/toggle`：切换 Base/LoRA
- `/base`：切到 Base
- `/lora`：切到 LoRA
- `/clear`：清屏
- `exit`：退出

说明：

- 当 LoRA 目录不存在或加载失败时，脚本会自动降级为 Base 模式，不会崩溃。

### 3) 批量对比

```bash
python src/compare.py \
   --model_name Qwen/Qwen2.5-1.5B-Instruct \
   --lora_path qwen_lora_output \
   --cache_dir local_cache \
   --max_new_tokens 256
```

说明：

- 若 LoRA 不可用，脚本会仅输出 Base 结果并给出提示。

## 关键现象与解释

在该配置下，常见现象包括：

- 中文问题中夹杂英文回答（语言分布偏移）
- 数学步骤变短甚至错误（推理链退化）
- 身份回答受历史蒸馏模板影响（角色漂移）

可将其视为数据分布与模型先验不匹配导致的负迁移案例。

## 常见问题

1. 显存不足

- 降低 `--batch_size`
- 增大 `--grad_accum`
- 降低 `--max_len`

2. Windows 多进程卡住

- 使用 `--num_proc 1 --num_workers 0`

3. 推理时报 LoRA 相关错误

- 检查 `--lora_path` 是否指向包含 `adapter_config.json` 的目录
- 若仅测试基座模型，可不提供有效 LoRA 目录

## 免责声明

本仓库用于研究与工程实践学习，不作为通用能力提升结论。不同数据质量、训练配方和超参数会显著影响结果。

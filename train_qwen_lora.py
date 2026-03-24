import os
import sys
import logging
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
# ✨ PEFT 库，用于 LoRA 微调
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

# ==========================================
# 1. 设置日志
# ==========================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Professional Qwen LoRA Training Script")

    # 路径配置
    parser.add_argument("--data_path", type=str, default="yahma/alpaca-cleaned", help="数据集")
    parser.add_argument("--cache_dir", type=str, default="./local_cache", help="缓存目录")
    parser.add_argument("--output_dir", type=str, default="./qwen_lora_output", help="输出目录")

    # 🔥 核心升级：默认使用 Qwen2.5-1.5B
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="显存不够就设为1")
    parser.add_argument("--grad_accum", type=int, default=16, help="梯度累积")
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA 的学习率")
    parser.add_argument("--max_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--eval_size", type=int, default=1000, help="验证集大小")
    parser.add_argument("--num_proc", type=int, default=1, help="dataset.map 进程数，Windows 建议 1")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers，Windows 建议 0")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 🌀 LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="防止过拟合")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 启动进阶训练流程，使用设备: {device}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # ==========================================
    # 2. 加载模型 (FP16 模式)
    # ==========================================
    logger.info(f"📥 加载 Qwen 模型: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    # ⚠️ 以半精度加载
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
        device_map=device_map
    )

    # 🛠️ 关键修改：在这里设置 use_cache = False
    # 这一步是为了兼容梯度检查点 (Gradient Checkpointing)
    model.config.use_cache = False

    # 开启梯度检查点
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # ==========================================
    # 3. 配置 LoRA
    # ==========================================
    logger.info("🔧 正在应用 LoRA 适配器配置...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ==========================================
    # 4. 数据处理 (Qwen ChatML 格式)
    # ==========================================
    logger.info(f"📥 加载并处理数据: {args.data_path}")
    train_dataset = load_dataset(args.data_path, split="train", cache_dir=args.cache_dir)

    if len(train_dataset) <= 1:
        raise ValueError("数据集样本过少，至少需要 2 条数据用于训练/验证切分。")

    eval_size = min(args.eval_size, max(1, len(train_dataset) // 10))
    if eval_size >= len(train_dataset):
        eval_size = len(train_dataset) - 1

    dataset = train_dataset.train_test_split(test_size=eval_size, seed=args.seed)

    def preprocess_function(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        model_inputs = []
        for i in range(len(instructions)):
            # Qwen ChatML 格式
            prompt_content = instructions[i]
            if inputs[i] and inputs[i].strip():
                prompt_content += f"\nContext: {inputs[i]}"

            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_content}<|im_end|>\n<|im_start|>assistant\n{outputs[i]}<|im_end|>"

            model_inputs.append(text)

        tokenized = tokenizer(
            model_inputs,
            truncation=True,
            max_length=args.max_len,
            padding="max_length"
        )

        input_ids = tokenized["input_ids"]
        labels = []
        for input_id_seq in input_ids:
            label_seq = [
                (token_id if token_id != tokenizer.pad_token_id else -100)
                for token_id in input_id_seq
            ]
            labels.append(label_seq)

        tokenized["labels"] = labels
        return tokenized

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=max(1, args.num_proc),
        remove_columns=dataset["train"].column_names
    )

    # ==========================================
    # 5. 训练参数
    # ==========================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),

        logging_dir=f"{args.output_dir}/logs",
        logging_steps=20,
        report_to="tensorboard",
        dataloader_num_workers=max(0, args.num_workers),

        # ❌ 已删除：use_cache=False (它不应该在这里)
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("🔥 开始 LoRA 微调 Qwen...")
    trainer.train()

    logger.info(f"✅ 训练完成！LoRA 适配器保存至: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
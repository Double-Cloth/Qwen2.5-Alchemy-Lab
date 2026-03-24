import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os
from contextlib import nullcontext

# =================配置区域=================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# ⚠️ 确保这里的路径是你刚刚训练生成的 checkpoint 文件夹，例如 checkpoint-200 或 final
LORA_PATH = "./qwen_lora_output"
CACHE_DIR = "./local_cache"


# =========================================
# 颜色代码，方便终端查看
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'  # 原版
    RED = '\033[91m'  # LoRA
    RESET = '\033[0m'


def parse_args():
    parser = argparse.ArgumentParser(description="Compare base model and LoRA model outputs")
    parser.add_argument("--model_name", type=str, default=BASE_MODEL, help="基础模型名称")
    parser.add_argument("--lora_path", type=str, default=LORA_PATH, help="LoRA 目录")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR, help="模型缓存目录")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成 token")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 采样")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚")
    return parser.parse_args()


def generate_response(model, tokenizer, prompt, args):
    # Qwen 推荐的 System Prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 裁剪掉 Prompt 部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def main():
    args = parse_args()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    print(f"{Colors.HEADER}⏳ 正在加载基础模型: {args.model_name}...{Colors.RESET}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. 先加载 Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    print(f"{Colors.HEADER}🔧 正在挂载 LoRA 适配器...{Colors.RESET}")
    # 2. 尝试挂载 LoRA
    has_lora = False
    model = base_model
    if os.path.exists(args.lora_path):
        try:
            model = PeftModel.from_pretrained(base_model, args.lora_path)
            has_lora = True
        except Exception as e:
            print(f"{Colors.RED}⚠️ LoRA 加载失败：{e}，将仅展示 Base 输出。{Colors.RESET}")
    else:
        print(f"{Colors.RED}⚠️ 未发现 LoRA 目录：{args.lora_path}，将仅展示 Base 输出。{Colors.RESET}")

    print("\n✅ 模型准备就绪！开始对比测试...\n")

    # 🔥 核心修改：专门设计的“照妖镜”问题集
    test_scenarios = [
        {
            "category": "🧐 身份认知 (Identity)",
            "prompt": "Who are you? Introduce yourself.",
            "expect": "原版会说我是Qwen；微调版可能会变得含糊或模仿Alpaca的回答。"
        },
        {
            "category": "🇨🇳 语言一致性 (Language Bias)",
            "prompt": "用中文介绍一下什么是由于'过拟合'导致的。",
            "expect": "原版中文流利；微调版极有可能因为Alpaca的英文权重过大，而突然崩出英文。"
        },
        {
            "category": "🧠 逻辑与风格 (Reasoning Style)",
            "prompt": "Solve this math problem: 25 - 4 * 2 + 3 = ?",
            "expect": "原版通常会一步步推理；Alpaca微调版可能会直接给答案，或者格式不同。"
        },
        {
            "category": "🐍 代码能力 (Coding)",
            "prompt": "Write a Python function to check if a number is prime.",
            "expect": "观察代码注释风格的变化。"
        }
    ]

    for item in test_scenarios:
        prompt = item["prompt"]
        category = item["category"]

        print("=" * 70)
        print(f"{Colors.BLUE}📌 测试维度: {category}{Colors.RESET}")
        print(f"❓ 问题: {prompt}")
        print("-" * 70)

        # --- A. 测试 原版模型 (Base) ---
        # 使用 disable_adapter() 上下文管理器，暂时“关掉”LoRA
        with (model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()):
            print(f"{Colors.GREEN}🟢 [Base Model (Qwen2.5)] 思考中...{Colors.RESET}")
            base_ans = generate_response(model, tokenizer, prompt, args)
            print(f"{Colors.GREEN}{base_ans}{Colors.RESET}")

        if has_lora:
            print("-" * 30 + " VS " + "-" * 30)

            # --- B. 测试 微调模型 (LoRA) ---
            # 上下文管理器结束，LoRA 自动重新生效
            print(f"{Colors.RED}🔴 [LoRA Model (Alpaca-Tuned)] 思考中...{Colors.RESET}")
            lora_ans = generate_response(model, tokenizer, prompt, args)
            print(f"{Colors.RED}{lora_ans}{Colors.RESET}")

        print("\n" + "." * 70 + "\n")


if __name__ == "__main__":
    main()

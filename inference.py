import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os

# ==========================
# ⚙️ 配置路径
# ==========================
# 基础模型（必须与训练时一致）
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# 训练好的 LoRA 保存目录
LORA_PATH = "./qwen_lora_output"
# 🔥 关键修改：指向你训练时使用的缓存目录，避免重新下载
CACHE_DIR = "./local_cache"


def load_model():
    print(f"⏳ 正在加载基础模型: {BASE_MODEL_NAME} ...")
    print(f"📂 使用缓存目录: {CACHE_DIR}")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        cache_dir=CACHE_DIR,  # 使用本地缓存
        trust_remote_code=True
    )

    # 2. 加载基础模型
    # device_map="auto" 会自动将模型分配到 GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        cache_dir=CACHE_DIR,  # 使用本地缓存
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. 加载 LoRA 适配器并与基础模型合并
    print(f"🔗 正在挂载 LoRA 适配器: {LORA_PATH} ...")

    if not os.path.exists(LORA_PATH):
        print(f"❌ 错误：找不到 LoRA 目录 '{LORA_PATH}'。请检查训练是否完成。")
        sys.exit(1)

    try:
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        print("✅ LoRA 加载成功！")
    except Exception as e:
        print(f"❌ 加载 LoRA 失败。错误信息: {e}")
        sys.exit(1)

    return model, tokenizer


def chat_loop(model, tokenizer):
    print("\n" + "=" * 50)
    print("🤖 Qwen-LoRA 交互模式 (输入 'exit' 退出)")
    print("=" * 50)

    # Qwen 的系统提示词
    system_prompt = "You are a helpful assistant."

    while True:
        try:
            query = input("\n👤 User: ").strip()
        except EOFError:
            break

        if not query:
            continue
        if query.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        # 构建对话
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 只截取新生成的回复部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"🤖 Assistant: {response}")


if __name__ == "__main__":
    model, tokenizer = load_model()
    chat_loop(model, tokenizer)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import os

# ==========================
# ⚙️ 配置路径
# ==========================
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./qwen_lora_output"
CACHE_DIR = "./local_cache"


# ==========================
# 🎨 颜色代码 (为了更好区分模式)
# ==========================
class Colors:
    LORA = '\033[91m'  # 红色 (代表微调版)
    BASE = '\033[92m'  # 绿色 (代表原版)
    USER = '\033[94m'  # 蓝色
    SYS = '\033[90m'  # 灰色
    RESET = '\033[0m'


def load_model():
    print(f"{Colors.SYS}⏳ 正在加载基础模型: {BASE_MODEL_NAME} ...{Colors.RESET}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"{Colors.SYS}🔗 正在挂载 LoRA 适配器: {LORA_PATH} ...{Colors.RESET}")

    has_lora = False
    if os.path.exists(LORA_PATH):
        try:
            # 加载 LoRA 模型
            model = PeftModel.from_pretrained(base_model, LORA_PATH)
            has_lora = True
            print(f"✅ LoRA 加载成功！")
        except Exception as e:
            print(f"❌ 加载 LoRA 失败: {e} (将仅使用基础模型)")
            model = base_model
    else:
        print(f"⚠️ 未找到 LoRA 文件，以纯基础模型模式运行。")
        model = base_model

    return model, tokenizer, has_lora


def generate_answer(model, tokenizer, query, use_lora):
    """
    统一生成函数，根据 use_lora 决定是否启用适配器
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 核心逻辑：根据开关决定是否使用 LoRA
    # model.disable_adapter() 是上下文管理器，在此缩进块内 LoRA 失效
    if use_lora:
        # LoRA 模式（PeftModel 默认就是启用的）
        context = torch.no_grad()  # 空的上下文，不起特殊作用
        mode_name = f"{Colors.LORA}[LoRA微调版]{Colors.RESET}"
    else:
        # Base 模式（强制关闭 LoRA）
        context = model.disable_adapter()
        mode_name = f"{Colors.BASE}[Base原版]{Colors.RESET}"

    print(f"{Colors.SYS}⚙️  正在生成 ({mode_name})...{Colors.RESET}")

    with torch.no_grad(), context:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def chat_loop(model, tokenizer, has_lora):
    print("\n" + "=" * 60)
    print("🤖 Qwen-LoRA 交互控制台")
    print(f"👉 输入 {Colors.LORA}/t{Colors.RESET} 或 {Colors.LORA}/toggle{Colors.RESET} 切换 [LoRA] / [Base] 模式")
    print(f"👉 输入 {Colors.LORA}/clear{Colors.RESET} 清屏")
    print(f"👉 输入 {Colors.LORA}exit{Colors.RESET} 退出")
    print("=" * 60)

    # 默认开启 LoRA (如果加载成功的话)
    current_mode_lora = has_lora

    while True:
        # 动态显示当前模式提示符
        status_icon = f"{Colors.LORA}🔥LoRA{Colors.RESET}" if current_mode_lora else f"{Colors.BASE}🧊Base{Colors.RESET}"

        try:
            query = input(f"\n👤 User ({status_icon}): ").strip()
        except EOFError:
            break

        if not query: continue

        # === 指令处理 ===
        cmd = query.lower()
        if cmd in ["exit", "quit"]:
            print("👋 Bye!")
            break

        if cmd == "/clear":
            print("\033c", end="")  # 清屏指令
            print(f"🧹 屏幕已清理")
            continue

        if cmd in ["/t", "/toggle", "切换"]:
            if not has_lora:
                print("❌ 无法切换：未加载 LoRA 模型。")
                continue
            current_mode_lora = not current_mode_lora
            new_status = "🔥 LoRA 微调模式" if current_mode_lora else "🧊 Base 原版模式"
            print(f"🔄 已切换为: {new_status}")
            continue

        if cmd in ["/base", "/b"]:
            current_mode_lora = False
            print(f"🔄 已切换为: 🧊 Base 原版模式")
            continue

        if cmd in ["/lora", "/l"]:
            if not has_lora:
                print("❌ 无法切换：未加载 LoRA 模型。")
                continue
            current_mode_lora = True
            print(f"🔄 已切换为: 🔥 LoRA 微调模式")
            continue

        # === 正常生成 ===
        response = generate_answer(model, tokenizer, query, use_lora=current_mode_lora)

        color = Colors.LORA if current_mode_lora else Colors.BASE
        print(f"🤖 Assistant: {color}{response}{Colors.RESET}")


if __name__ == "__main__":
    # 加载时返回一个标志位，告诉主循环是否有 LoRA 可用
    model, tokenizer, has_lora = load_model()
    chat_loop(model, tokenizer, has_lora)

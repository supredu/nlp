import torch
from transformers import AutoTokenizer
from model.model import MiniMindLM
from model.config import LMConfig

# === 1. 模型配置 ===
config = LMConfig(
    dim=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=2,
    vocab_size=6400,  # 确保与 tokenizer 一致
    model_max_length=512,
)

# === 2. 加载 tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")

# === 3. 加载模型权重 ===
model = MiniMindLM(config)
checkpoint = torch.load("out/sft_512.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# === 4. 准备输入并生成翻译 ===
prompt = "Translate to Traditional Chinese: The stock market crashed today due to unexpected inflation data."
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9
    )

# === 5. 打印输出 ===
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🧪 Prompt:", prompt)
print("📘 Output:", result)

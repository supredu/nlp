#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# 避免 HuggingFace tokenizers 警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 将项目根目录加入 PYTHONPATH，确保能导入 model 包
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from model.model import MiniMindLM
from model.config import LMConfig
from openai import OpenAI

# === 配置 ===
SENTENCES_FILE = "data/lora.jsonl"   # 输入的金融英文句子
OUTPUT_FILE    = "data/dpo.jsonl"    # 输出的偏好对
MAX_PAIRS      = 2000
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_RETRIES    = int(os.getenv("DPO_MAX_RETRIES", "3"))

# === DeepSeek 客户端 ===
DS_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DS_KEY:
    raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")
client = OpenAI(api_key=DS_KEY, base_url="https://api.deepseek.com")

# === 加载 SFT 模型 & tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(ROOT / "model" / "tokenizer")
config    = LMConfig()
model     = MiniMindLM(config).to(DEVICE).eval()

def generate_variants(en_sentence: str):
    """
    用两套采样参数生成 A/B 两个繁体中文译文（去掉英文 prompt）。
    """
    prompt = f"Translate to Traditional Chinese: {en_sentence}"
    enc = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]

    # Variant A
    out1 = model.generate(**enc, temperature=0.7, top_p=0.9, max_new_tokens=128)
    # Variant B
    out2 = model.generate(**enc, temperature=1.0, top_p=0.95, max_new_tokens=128)

    # 只保留生成的新 token（去掉原始 prompt 部分）
    gen_ids_a = out1[0][prompt_len:]
    gen_ids_b = out2[0][prompt_len:]

    a = tokenizer.decode(gen_ids_a, skip_special_tokens=True).strip()
    b = tokenizer.decode(gen_ids_b, skip_special_tokens=True).strip()
    return a, b

def query_deepseek_choice(en: str, a: str, b: str) -> str:
    """
    调用 DeepSeek 比较 A/B 译文，返回 '1' 或 '2'。
    限流重试，失败后默认 '1'。
    """
    prompt = (
        f"Compare these two Traditional Chinese translations for the English sentence:\n\n"
        f"\"{en}\"\n\n"
        f"1) {a}\n"
        f"2) {b}\n\n"
        "Reply with only the number (1 or 2) indicating which is more accurate and natural."
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4,
                temperature=0.0,
                stream=False
            )
            choice_text = resp.choices[0].message.content.strip()
            return choice_text[0] if choice_text and choice_text[0] in ("1", "2") else "1"
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RateLimit" in msg:
                wait = 5 * attempt
                print(f"[Warning] Rate limit (attempt {attempt}), retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"[Error] DeepSeek error: {e}, defaulting to '1'", file=sys.stderr)
            return "1"
    print("[Warning] Max retries exceeded, default to '1'", file=sys.stderr)
    return "1"

def main():
    with open(SENTENCES_FILE, encoding="utf-8") as fin:
        en_list = [json.loads(line)["en"] for line in fin]

    written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for en in tqdm(en_list[:MAX_PAIRS], desc="Generating DPO pairs"):
            a, b = generate_variants(en)
            choice = query_deepseek_choice(en, a, b)
            if choice == "2":
                chosen, rejected = b, a
            else:
                chosen, rejected = a, b

            fout.write(json.dumps({"x": en,       "chosen": chosen},  ensure_ascii=False) + "\n")
            fout.write(json.dumps({"x": en,       "rejected": rejected}, ensure_ascii=False) + "\n")
            written += 1

    print(f"[Done] Wrote {written} DPO pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

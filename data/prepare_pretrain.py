# data/prepare_pretrain.py

from datasets import load_dataset
from opencc import OpenCC
import json

def build_pretrain_jsonl(output_path: str, limit: int = 50000):
    # 加载 OPUS-100 英–简体中文对照（config 名称 "en-zh"）
    ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="train")
    cc = OpenCC('s2t')  # 简体→繁体
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, item in enumerate(ds):
            if i >= limit:
                break
            en = item["translation"]["en"]
            zh_simp = item["translation"]["zh"]
            zh_trad = cc.convert(zh_simp)
            fout.write(json.dumps({"en": en, "zh": zh_trad}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    build_pretrain_jsonl("data/pretrain.jsonl")
    print("Generated data/pretrain.jsonl")

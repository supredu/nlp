from datasets import load_dataset
from opencc import OpenCC
import json

def build_sft_jsonl(output_path, limit=20000):
    ds = load_dataset("wmt17", "zh-en", split="train")
    cc = OpenCC('s2t')
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, item in enumerate(ds):
            if i >= limit:
                break
            en = item["translation"]["en"]
            zh_trad = cc.convert(item["translation"]["zh"])
            instruction = f"Translate to Traditional Chinese: {en}"
            fout.write(json.dumps({
                "instruction": instruction,
                "response": zh_trad
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    build_sft_jsonl("data/sft.jsonl")
    print("Generated data/sft.jsonl")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, json
from opencc import OpenCC

# 提升单元格大小上限
import csv
csv.field_size_limit(sys.maxsize)


def build_lora_jsonl(csv_path: str, output_path: str, limit: int = 20000):
    """
    从 CSV（ASCII 分号分隔）读取每行：
    id;date;title_en;title_zh;flag;content_en;content_zh
    按分号最多切 6 次，取第 6/7 段为正文英/中，再转繁体，写 JSONL。
    """
    cc = OpenCC('s2t')
    written = 0

    with open(csv_path, encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            # 跳过空行
            if not line.strip():
                continue

            # 最多 split 6 次，得到 7 段
            parts = line.rstrip('\n').split(';', 6)
            if len(parts) < 7:
                continue

            en = parts[5].strip()
            zh_simp = parts[6].strip()
            # 繁体转换
            zh_trad = cc.convert(zh_simp)

            # 写出 JSONL
            record = {'en': en, 'zh': zh_trad}
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

            written += 1
            if written >= limit:
                break

    print(f"Wrote {written} pairs to {output_path}")


if __name__ == '__main__':
    build_lora_jsonl(
        csv_path='data/FT-en-zh.csv',
        output_path='data/lora.jsonl',
        limit=20000
    )

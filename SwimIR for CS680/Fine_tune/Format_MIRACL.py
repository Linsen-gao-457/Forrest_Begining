from datasets import load_dataset
import json
import os
from tqdm import tqdm

output_path = "miracl_all_dev.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    languages = [
        "ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja",
        "ko", "ru", "sw", "te", "th", "yo", "zh"
    ]

    slice_percent = 1
    for lang in languages:
        print(f"Processing language: {lang}")
        for i in range(0, 100, slice_percent):
            try:
                print(f"  - Loading slice {i}% to {i+slice_percent}%")
                dataset = load_dataset(
                    "miracl/miracl", name=lang,
                    split=f"dev[{i}%:{i+slice_percent}%]",
                    trust_remote_code=True
                )
                for item in tqdm(dataset, desc=f"{lang} [{i}%]", leave=False):
                    entry = {
                        "language": lang,
                        "query_id": item["query_id"],
                        "query": item["query"],
                        "positive_passages": item["positive_passages"],
                        "negative_passages": item["negative_passages"]
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed {lang} [{i}%:{i+slice_percent}%]: {e}")

print(f"Finished writing to {output_path}")
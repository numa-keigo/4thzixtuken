import os

# データセットがキャッシュ済みの場合
os.environ["HF_DATASETS_OFFLINE"] = "1"

#komenn
import csv
import ollama
from datasets import load_dataset


ds = load_dataset(
     "HuggingFaceH4/MATH-500",
     cache_dir="./hf_datasets",
 )

with open(
    "qwen3_answer.csv",
    mode="w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow(["id", "problem", "solution", "llm_answer"])

    for i in range(100):
        problem = ds["test"][i]["problem"]
        solution = ds["test"][i]["solution"]

        response = ollama.chat(
            model="qwen3:8b",
            messages=[
                {"role": "user", "content": problem}
            ],
            think=False,
        )

        llm_answer = response["message"]["content"]

        writer.writerow([
            i + 1,
            problem,
            solution,
            llm_answer,
        ])
        f.flush()

        print(f"{i + 1}問目終了！", flush=True)

print("終了！")
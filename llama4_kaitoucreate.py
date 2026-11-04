import ollama
import os
import csv

# オフラインモード
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


from datasets import load_dataset
#環境確認
print("環境確認")
print(os.environ.get("TRANSFORMERS_OFFLINE"))
print(os.environ.get("HF_DATASETS_OFFLINE"))

# データセット読み込み
ds = load_dataset("HuggingFaceH4/MATH-500", cache_dir="./hf_datasets")

with open("llama4_answer.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # ヘッダー行
    writer.writerow(["id","problem", "solution","llm_answer"])

    for i in range(100):

        #番号
        id = str(i+1)
        
        #問題
        problem = ds["test"][i]["problem"]
        
        #模範解答
        solution = ds["test"][i]["solution"]
        # LLM の出力
        response = ollama.chat(
            model="llama4",
            messages=[{"role": "user", "content": problem}]
        )

        llm_answer = response["message"]["content"]
        #ファイルに書く
        writer.writerow([id,problem, solution, llm_answer])
        print(str(i+1),"問目終了！")

print("終了！！！！！！")
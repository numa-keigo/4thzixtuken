import csv
import os
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset




# データセット読み込み
ds = load_dataset("HuggingFaceH4/MATH-500", cache_dir="./hf_datasets")

# chat gpt を呼び出し
#load
load_dotenv()
 # 環境変数からAPIキーを読み込む
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# 保存用 CSV ファイルを開く
with open("chatgpt_answer.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # ヘッダー行
    writer.writerow(["id","problem", "solution", "llm_answer"])


    for i in range(100):

        #番号
        id = str(i+1)
        
        #問題
        problem = ds["test"][i]["problem"]
        problem_solution = ds["test"][i]["solution"]
        #chatgptに聞く
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a helpful assistant.Please explain the solution process carefully. "},
                {"role": "user", "content":f"Please solve the following problem.：{problem}"}
            ]
        )
        
        llm_answer = response.choices[0].message.content
             
        writer.writerow([id , problem , problem_solution , llm_answer])
        print(str(i+1),"問目終了！")
#ラスト閉め
print("終了！！！！")

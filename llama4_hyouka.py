import sacrebleu
from bert_score import score
import csv
import pandas as pd
import os

# オフラインモード
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


from sentence_transformers import SentenceTransformer, util

#モデルの読み込み
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

df = pd.read_csv("llama4_answer.csv")


# 保存用 CSV ファイルを開く
with open("llama4_hyouka.csv", mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # ヘッダー行
    writer.writerow(["id", "bleuscore","Precision:", "Recall", "F1","similarity"])
    for i in range(100):

        #idを定義
        id = str(i+1)
        
        #llmの解答を定義
        hyp = df["llm_answer"].iloc[i]

        #模範解答を定義
        ref = [df["solution"].iloc[i]]
        
        #bleuscoreを計算        
        bleu = sacrebleu.sentence_bleu(hyp, ref)
        
        bleu_score = bleu.score
        
        ########################################################################################################
        
        
        #bertscoreを計算
        
        cands = [df["llm_answer"].iloc[i]]
        
        P, R, F1 = score(
            cands, ref,
            lang="en",
            model_type="roberta-large"
        )
        #####################################################################################################
        
        #コサイン類似度を計算
        emb1 = model.encode(df["solution"].iloc[i], convert_to_tensor=True)
        emb2 = model.encode(df["llm_answer"].iloc[i], convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
        
        ##################################################################################################

        writer.writerow([id, bleu_score, P.item(),R.item(),F1.item(), similarity])
        print(str(i+1),"問目終了！")
print("終了")
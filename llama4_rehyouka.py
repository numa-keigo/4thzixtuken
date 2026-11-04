import os

# 関連ライブラリのimportより先に設定
# 必要なモデルがダウンロード済みの場合に使う
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv
import pandas as pd
import sacrebleu
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util


# モデルはそれぞれ一度だけ読み込む
sentence_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

bert_scorer = BERTScorer(
    lang="en",
    model_type="roberta-large",
)

df = pd.read_csv("llama4_answer.csv")

# 元の処理と同じく、先頭100問を対象にする
if len(df) < 100:
    raise ValueError("入力CSVには100問以上必要です。")

target = df.iloc[:100]

# 空欄や数値が入っている場合は、計算前に確認する
for column in ["llm_answer", "solution"]:
    if not target[column].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError(f"{column}に空欄または文字列以外があります。")

with open(
    "llama4_hyouka.csv",
    mode="w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "bleuscore", "Precision", "Recall", "F1", "similarity"
    ])

    for i in range(100):
        hyp = target["llm_answer"].iloc[i]
        solution = target["solution"].iloc[i]
        refs = [solution]

        # BLEU
        bleu_score = sacrebleu.sentence_bleu(hyp, refs).score

        # BERTScore：読み込み済みモデルを使う
        P, R, F1 = bert_scorer.score([hyp], refs)

        # Sentence-BERTによるコサイン類似度
        emb1 = sentence_model.encode(
            solution, convert_to_tensor=True
        )
        emb2 = sentence_model.encode(
            hyp, convert_to_tensor=True
        )
        similarity = util.cos_sim(emb1, emb2).item()

        writer.writerow([
            i + 1,
            bleu_score,
            P.item(),
            R.item(),
            F1.item(),
            similarity,
        ])
        f.flush()

        print(f"{i + 1}問目終了！", flush=True)

print("終了")
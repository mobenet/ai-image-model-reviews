from transformers import pipeline 
import pandas as pd 
import torch

device = 0 if torch.cuda.is_available() else -1

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device
)

df = pd.read_csv("data/ai_image_reviews_1500.csv")


df["sentiment_result"] = df["text"].astype(str).apply(lambda x: sentiment_pipeline(x[:512])[0])
df["sentiment_label"] = df["sentiment_result"].apply(lambda x: x["label"])
df["sentiment_score"] = df["sentiment_result"].apply(lambda x: x["score"])

df.drop(columns=["sentiment_result"], inplace=True)
df.to_csv("data/ai_image_reviews_1500_sentiment.csv", index=False)


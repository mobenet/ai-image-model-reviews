from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import torch
tqdm.pandas() 

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("text2text-generation", model="google/long-t5-tglobal-base")

df = pd.read_csv("data/ai_image_reviews_1500.csv")

df["summarization"] = df["text"].astype(str).progress_apply(lambda x: summarizer(x, max_new_tokens=512, min_new_tokens=30)[0]["generated_text"])
df.to_csv("data/ai_image_reviews_1500_summarization.csv", index=False)




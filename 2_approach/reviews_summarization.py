from transformers import pipeline, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import torch
tqdm.pandas() 

# device = 0 if torch.cuda.is_available() else -1
# summarizer = pipeline("text2text-generation", model="google/long-t5-tglobal-base")

# df = pd.read_csv("data/ai_image_reviews_1500.csv")

# df["summarization"] = df["text"].astype(str).progress_apply(lambda x: summarizer(x, max_new_tokens=512, min_new_tokens=30)[0]["generated_text"])
# df.to_csv("data/ai_image_reviews_1500_summarization.csv", index=False)



df = pd.read_csv("data/ai_image_model_reviews_2.csv")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

def has_more_512(text):
    tokens = tokenizer.encode(str(text), truncation=False)
    return len(tokens) > 512

df["over_512"] = df["text"].apply(has_more_512)

under_512 = df[~df["over_512"]]

under_512.to_csv("data/ai_image_reviews_under_512.csv", index=False)
print(f"Nombre de files amb 512 o menys tokens: {len(under_512)}")
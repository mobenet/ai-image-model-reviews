import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time, os
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

INPUT_PATH = "2_approach/ai_image_reviews_t5_large_clean_summarization.csv"
OUTPUT_PATH = "2_approach/reviews_openai_analysis_partial.csv"

# Carrega el CSV original o el parcial si ja existeix
if os.path.exists(OUTPUT_PATH):
    print(f"Loading progress from: {OUTPUT_PATH}")
    df = pd.read_csv(OUTPUT_PATH)
else:
    print(f"Starting new process from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    df["openai_analysis"] = None

# Funció per saber si una fila ja ha estat processada
def is_processed(x):
    return isinstance(x, str) and x.strip().startswith("{") and len(x.strip()) > 10

# Funció per fer la classificació amb GPT
def classify_review(text):
    prompt = f"""
Given the following customer review, classify it as 'Good', 'Neutral', or 'Bad'.
Also extract 2-3 key points from the review that summarize its main ideas.

Review:
\"\"\"{text}\"\"\"

Return your response as a JSON object like this:
{{
  "sentiment": "Good",
  "key_points": ["...", "..."]
}}
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Error:", e)
        return None

# Recorrem les files pendents
for i in tqdm(range(len(df))):
    if is_processed(df.loc[i, "openai_analysis"]):
        continue  # salta si ja estava fet

    review = str(df.loc[i, "summarization"])
    result = classify_review(review)
    df.loc[i, "openai_analysis"] = result
    time.sleep(1.2)

    # Desa cada 10 files
    if i % 10 == 0 or result is None:
        df.to_csv(OUTPUT_PATH, index=False)

# Desa el resultat final
df.to_csv(OUTPUT_PATH, index=False)
print("Procés completat o quota exhaurida. Resultats desats.")

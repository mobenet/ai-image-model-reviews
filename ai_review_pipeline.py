from scraping.reddit_scraper import RedditReviewScraper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline, AutoTokenizer
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import re, time, os, html, json, torch
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

"""Scraping"""
scraper = RedditReviewScraper(
    reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
    reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

suggestions = scraper.get_llm_suggestions("AI image generation models")
print(suggestions)

subreddits = suggestions["subreddits"]
products = suggestions["products"]
search_terms = suggestions["search_terms"]

df = scraper.scrape_reviews(
    category_type="AI image generation models",
    custom_subreddits=subreddits,
    custom_products=["Midjourney",
                    "DALL·E",
                    "Stable Diffusion",
                    "Leonardo AI",
                    "Runway ML"],
    custom_search_terms=search_terms
)


"""Summarization"""
summarizer = pipeline("text2text-generation", model="google-t5/t5-large")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

def clean(text):
    text = html.unescape(text)  
    text = re.sub(r'https?://\S+', '', text)  
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text) 
    text = re.sub(r'[^\w\s.,!?\'\"]+', '', text)  
    text = re.sub(r'\s+', ' ', text) 
    return text.strip()

def summarize_text(text):
  clean_text = clean(text)
  tokenized = tokenizer.encode(clean_text, truncation=False)
  if len(tokenized) > 512:
    try:
      return summarizer(clean_text, max_new_tokens=300, min_new_tokens=30, no_repeat_ngram_size=3)[0]["generated_text"]
    except Exception as e:
      print(f"Error summarizing: {e}")
      return clean_text
  else:
    return clean_text

df["summarization"] = df["text"].astype(str).progress_apply(summarize_text)
df.to_csv("ai_image_reviews_t5_large_clean_summarization.csv", index=False)


"""Classification"""
INPUT_PATH = "ai_image_reviews_t5_large_clean_summarization.csv"
OUTPUT_PATH = "reviews_openai_analysis_partial.csv"

if os.path.exists(OUTPUT_PATH):
    print(f"Loading progress from: {OUTPUT_PATH}")
    df = pd.read_csv(OUTPUT_PATH)
else:
    print(f"Starting new process from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    df["openai_analysis"] = None

def is_processed(x):
    return isinstance(x, str) and x.strip().startswith("{") and len(x.strip()) > 10

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

for i in tqdm(range(len(df))):
    if is_processed(df.loc[i, "openai_analysis"]):
        continue 

    review = str(df.loc[i, "summarization"])
    result = classify_review(review)
    df.loc[i, "openai_analysis"] = result
    time.sleep(1.2)

    if i % 10 == 0 or result is None:
        df.to_csv(OUTPUT_PATH, index=False)

df.to_csv(OUTPUT_PATH, index=False)


"""Clustering"""
df = df.dropna(subset=['openai_analysis'])
df.drop(['category_type', 'search_term', 'title', 'date', 'score', 'comments', 'subreddit', 'url', 'Unnamed: 10', 'Unnamed: 11', 'summarization'], axis=1, inplace=True)

df['openai_analysis'] = df['openai_analysis'].apply(json.loads)
df_analisis = pd.json_normalize(df['openai_analysis'])
df = df.join(df_analisis)
df.drop(['openai_analysis'], axis=1, inplace=True)

df['sentiment'] = df['sentiment'].str.lower()
df = df[df['sentiment'].isin(['good', 'bad'])]

df['key_points'] = df['key_points'].apply(lambda kp: ' '.join(kp))
grouped = df.groupby(['product', 'sentiment'])['key_points'].apply(lambda texts: ' '.join(texts)).reset_index()



vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
tfidf_matrix = vectorizer.fit_transform(grouped['key_points'])

n_topics = 3
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_model.fit(tfidf_matrix)
topic_distribution = lda_model.transform(tfidf_matrix)
grouped['dominant_topic'] = topic_distribution.argmax(axis=1)

"""Review generator"""

def get_topic_keywords(lda_model, vectorizer, n_words=10):
    topic_keywords = {}
    feature_names = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topic_keywords[idx] = top_words
    return topic_keywords


def generate_prompts(grouped_df, lda_model, vectorizer):
    topic_keywords = get_topic_keywords(lda_model, vectorizer)
    prompts = {}

    providers = grouped_df['product'].unique()

    for provider in providers:
        entry = grouped_df[grouped_df['product'] == provider]
        good_topic = entry[entry['sentiment'] == 'good']['dominant_topic'].values
        bad_topic = entry[entry['sentiment'] == 'bad']['dominant_topic'].values

        good_words = topic_keywords[good_topic[0]] if len(good_topic) else []
        bad_words = topic_keywords[bad_topic[0]] if len(bad_topic) else []

        prompt = f"""
You are a product review writer. Based on user feedback, summarize the main experiences people have with {provider}.

Positive reviews mention topics such as: {', '.join(good_words)}.

Negative reviews focus on: {', '.join(bad_words)}.

Write a short article recommending when to use {provider}, what to expect, and what potential issues to consider.
        """.strip()

        prompts[provider] = prompt

    return prompts

def generate_articles_from_prompts(prompts, model="gpt-4", max_tokens=400):
    results = {}

    for provider, prompt in prompts.items():
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful and professional product reviewer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            results[provider] = response.choices[0].message.content
        except Exception as e:
            results[provider] = f"Error: {e}"

    return results

prompts = generate_prompts(grouped, lda_model, vectorizer)
articles = generate_articles_from_prompts(prompts)
for provider, text in articles.items():
    with open(f"{provider.replace(' ', '_')}_article.txt", "w", encoding="utf-8") as f:
        f.write(text)
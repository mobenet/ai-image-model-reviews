from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd


df = pd.read_csv("./1_approach/balanced_2000_reviews.csv")

texts = df["text"].astype(str).tolist()

#model compacte i ràpid per transformar textos en vectors semàntics.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#converteix cada review en un vector que representa el significat del text
embeddings = embedding_model.encode(texts, show_progress_bar=True)
# agrupem textos en 6 temes maxim 
topic_model = BERTopic(nr_topics=6) 


topics, probs = topic_model.fit_transform(texts, embeddings)

df["topic"] = topics
df.to_csv("./1_approach/balanced_with_topics.csv", index=False)
print(topic_model.get_topic_info())
topic_model.visualize_topics()


for i in range(6):
    print(f"\nTOPIC {i}")
    print(df[df['topic'] == i]['text'].sample(3).values)
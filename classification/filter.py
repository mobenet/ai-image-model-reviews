import pandas as pd

# Carrega el dataset
df = pd.read_csv("data/ai_image_reviews_19k_sentiment.csv")

# Productes i sentiments que volem incloure
target_products = ["Midjourney", "Stable Diffusion", "Runway ML", "Leonardo AI", "DALL·E"]
sentiments = ["positive", "neutral", "negative"]

# Nombre desitjat per producte i sentiment
samples_per_group = 400 // len(sentiments)

# Filtra només les files amb els productes objectiu
df = df[df["product"].str.lower().isin(target_products)]

# Inicialitza una llista per guardar els grups seleccionats
balanced_rows = []

# Itera per producte i sentiment, seleccionant mostres
for product in target_products:
    for sentiment in sentiments:
        group = df[
            (df["product"].str.lower() == product)
            & (df["sentiment_label"].str.lower() == sentiment)
        ]
        if len(group) >= samples_per_group:
            sampled = group.sample(n=samples_per_group, random_state=42)
        else:
            # Agafa el màxim disponible si no n'hi ha prou
            sampled = group
        balanced_rows.append(sampled)

# Concatena tots els grups seleccionats
balanced_df = pd.concat(balanced_rows).sample(frac=1, random_state=42).reset_index(drop=True)

# Guarda el nou CSV
balanced_df.to_csv("balanced_2000_reviews.csv", index=False)

print(f"Dataset generat amb {len(balanced_df)} files a 'balanced_2000_reviews.csv'")

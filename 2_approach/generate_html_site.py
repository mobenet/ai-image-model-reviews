from pathlib import Path

def build_html(grouped_df, topic_keywords, articles, output_path="index.html"):
    html = ["""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Image Generation Models Review</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; background-color: #f4f4f4; }
            h1, h2 { color: #333; }
            .product { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
            .topic-list { font-style: italic; color: #555; }
            .article { margin-top: 10px; background: #fafafa; padding: 10px; border-left: 4px solid #0077cc; }
            .section-title { font-weight: bold; margin-top: 15px; }
        </style>
    </head>
    <body>
        <h1>Image Generation Models Comparison</h1>
        <p>This site presents an automatic analysis of user reviews on different AI image generation models, using sentiment classification, topic modeling (LDA), and text generation with OpenAI.</p>
    """]

    for product in grouped_df['product'].unique():
        entry = grouped_df[grouped_df['product'] == product]
        sentiments = {}

        for sentiment in ['good', 'bad']:
            dominant = entry[entry['sentiment'] == sentiment]['dominant_topic']
            if not dominant.empty:
                sentiments[sentiment] = topic_keywords[dominant.values[0]]

        summary = ""
        if sentiments.get('good'):
            summary += f"Positive topic keywords: <span class='topic-list'>{', '.join(sentiments['good'])}</span><br>"
        if sentiments.get('bad'):
            summary += f"Negative topic keywords: <span class='topic-list'>{', '.join(sentiments['bad'])}</span>"

        article = articles.get(product, "No article generated.")

        html.append(f"""
        <div class='product'>
            <h2>{product}</h2>
            <div class='section-title'>Sentiment Topics:</div>
            <p>{summary}</p>
            <div class='section-title'>Generated Article:</div>
            <div class='article'>
                <p>{article}</p>
            </div>
        </div>
        """)

    html.append("</body></html>")

    Path(output_path).write_text('\n'.join(html), encoding='utf-8')
    print(f"HTML file generated at: {output_path}")

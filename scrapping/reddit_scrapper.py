import praw
import pandas as pd
from datetime import datetime
from openai import OpenAI
from typing import List, Dict, Optional
import os
import time
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
REDDIT_CLIENT_ID  = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET  = os.getenv('REDDIT_CLIENT_SECRET')

class RedditReviewScraper:
    def __init__(self, reddit_client_id: str, reddit_client_secret: str, openai_api_key: str):
        """Initialize the scraper with necessary API credentials."""
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret
        self.openai_api_key = openai_api_key
        """Initialize reddit."""
        self.reddit = praw.Reddit(
            client_id=self.reddit_client_id,
            client_secret=self.reddit_client_secret,
            username=os.getenv('REDDIT_USERNAME'),
            password=os.getenv('REDDIT_PASSWORD'),
            user_agent='windows:ai_image_review:v0.1 (by u/Scary-Buy7940)'
        )
        self.openai_client = OpenAI(api_key=self.openai_api_key)


    def get_llm_suggestions(self, category_type: str) -> Dict[str, List[str]]:

        """Use LLM to suggest relevant subreddits and search terms."""
        prompt = f"""
        For the product category "{category_type}", please provide:
        1. A list of 5-10 relevant subreddits where users might discuss these products
        2. A list of 5-10 specific products or brands in this category
        3. A list of 5-10 related search terms or keywords

        Format your response as a Python dictionary with three keys:
        'subreddits', 'products', 'search_terms'
        Each value should be a list of strings.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides structured data for product research."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
        # Parse the response - assuming it's in the correct format
            return eval(response.choices[0].message.content)
        except Exception as e:
            print(f"Error using GPT: {e}")
            # Fallback to default values if parsing fails
            # return {
            #     'subreddits': ['reviews', 'BuyItForLife', 'shopping'],
            #     'products': [category_type],
            #     'search_terms': [f'{category_type} review']
            # }

    def scrape_reviews(
        self,
        category_type: str,
        time_filter: str = 'year',
        limit: int = 100,
        custom_subreddits: Optional[List[str]] = None,
        custom_products: Optional[List[str]] = None,
        custom_search_terms: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Scrape reviews for a given product category."""

        # Get suggestions from LLM if custom values aren't provided
        if not all([custom_subreddits, custom_products, custom_search_terms]):
            suggestions = self.get_llm_suggestions(category_type)
            print("GPT suggestions: ", suggestions)
            subreddits = custom_subreddits or suggestions['subreddits']
            products = custom_products or suggestions['products']
            search_terms = custom_search_terms or suggestions['search_terms']
        else:
            subreddits = custom_subreddits
            products = custom_products
            search_terms = custom_search_terms

        reviews = []

        for product in products:
            for subreddit_name in subreddits:
                for search_term in search_terms:
                    try:
                        print(f"Searching {subreddit_name} for {product} using term: {search_term}")
                        subreddit = self.reddit.subreddit(subreddit_name)

                        search_query = f"{product} {search_term}"
                        for post in subreddit.search(search_query, time_filter=time_filter, limit=limit):
                            reviews.append({
                                'category_type': category_type,
                                'product': product,
                                'search_term': search_term,
                                'title': post.title,
                                'text': post.selftext,
                                'date': datetime.fromtimestamp(post.created_utc),
                                'score': post.score,
                                'comments': post.num_comments,
                                'subreddit': subreddit_name,
                                'url': f"https://reddit.com{post.permalink}"
                            })
                            print('SUCCESS')
                            time.sleep(1)
                    except Exception as e:
                        print(f"Error searching {subreddit_name}: {str(e)}")
                        print(f"reviews {subreddit_name}: {str(e)}")

        # Create DataFrame
        df = pd.DataFrame(reviews) if reviews else pd.DataFrame()

        # Print summary statistics
        if not df.empty:
            print(f"\nCollected {len(reviews)} reviews")
            print("\nReviews by product:")
            print(df['product'].value_counts())

        return df

def main():
    scraper = RedditReviewScraper(
        reddit_client_id=REDDIT_CLIENT_ID,
        reddit_client_secret=REDDIT_CLIENT_SECRET,
        openai_api_key=OPENAI_API_KEY
    )

    # Example 1: Fully automated with LLM suggestions
    # df = scraper.scrape_reviews(category_type="AI image generation models like Midjourney, DALL·E and Stable Diffusion")
    # df.to_csv('ai_image_model_reviews_1.csv', index=False)

    # Example 2: Custom parameters
    # custom_df = scraper.scrape_reviews(
    #     category_type="AI image generation models",
    #     custom_subreddits=['MachineLearning', 'DeepLearning', 'ArtificialInteligence', 
    #                         'computervision', 'learnmachinelearning', 'DataScience', 
    #                         'Python', 'AI', 'NeuralNetworks', 'Generative'],
    #     custom_products=['Midjourney', 'DALL·E', 'Stable Diffusion', 'DeepArt', 'DeepDream'],
    #     custom_search_terms=['quality', 'ethics', 'pricing', 'control', 'style','review']
    # )

    custom_df = scraper.scrape_reviews(
        category_type="AI image generation models",
        
        custom_subreddits=[
            "StableDiffusion",    # actiu i amb usuaris avançats
            "Midjourney",         # comunitat molt gran
            "aiArt",              # enfoc en art generat per IA
            "generative",         # més tècnic però útil
            "ArtificialInteligence",  # ocasionalment reviews o comparacions
            "Dalle2",             # subreddits específics de DALL·E
            "DeepDream",          # antics però encara hi ha debat
            "RunwayML"            # plataforma d'ús general
        ],

        custom_products=[
            "Midjourney",
            "DALL·E",
            "Stable Diffusion",
            "Leonardo AI",
            "Dream by Wombo",
            "Runway ML",
            "Deep Dream",
            "Artbreeder",
            "NightCafe",
            "StarryAI"
        ],

        custom_search_terms=[
            "review",
            "my experience",
            "hands-on",
            "comparison",
            "tried",
            "tested",
            "using",
            "workflow",
            "prompting",
            "best settings",
            "first impressions",
            "AI art workflow",
            "vs Midjourney",
            "vs DALL·E",
            "how to use",
            "what I got",
            "opinion",
            "performance",
            "output quality"
        ]
    )
    custom_df.to_csv('ai_image_model_reviews_2.csv', index=False)

if __name__ == "__main__":
    main()

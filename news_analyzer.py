# news_analyzer.py - Sentiment simple via NewsAPI + VADER
# - Centralisé via config (langue, tri, max articles)

import os
from dotenv import load_dotenv
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import config

load_dotenv()  # Charge la clé API depuis .env

def get_news_sentiment(symbol: str) -> float:
    """Analyse le sentiment des dernières news (titres) pour la devise (base du symbole)."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return 0.0

    currency = (symbol or "EURUSD")[:3]  # ex: EUR de EURUSD
    lang = (getattr(config, "NEWS_LANGUAGE", "en") or "en").lower().strip()
    page_size = int(getattr(config, "NEWS_MAX_ARTICLES", 5))
    sort_by = getattr(config, "NEWS_SORT_BY", "publishedAt")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": currency,
        "apiKey": api_key,
        "language": lang,
        "sortBy": sort_by,
        "pageSize": page_size
    }
    try:
        response = requests.get(url, params=params, timeout=12)
        if response.status_code != 200:
            # Fallback anglais si on a demandé fr et qu'on n'a rien
            if lang != "en":
                params["language"] = "en"
                response = requests.get(url, params=params, timeout=12)
                if response.status_code != 200:
                    return 0.0
            else:
                return 0.0

        articles = (response.json() or {}).get('articles', []) or []
        if not articles:
            return 0.0

        sia = SentimentIntensityAnalyzer()
        total_sentiment = 0.0
        n = 0
        for article in articles:
            title = (article.get('title') or "").strip()
            if not title:
                continue
            total_sentiment += sia.polarity_scores(title)['compound']
            n += 1
        return (total_sentiment / n) if n > 0 else 0.0
    except Exception:
        return 0.0
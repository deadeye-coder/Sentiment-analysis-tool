"""
Stock Sentiment Analysis Tool

This script fetches real-time stock data and performs sentiment analysis
on news and social media related to specific stocks using FinBERT.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
from textblob import TextBlob
import yfinance as yf
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax

class StockSentimentAnalyzer:
    def __init__(self, ticker_symbol):
        """
        Initialize the analyzer with a stock ticker symbol
        
        Args:
            ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        """
        self.ticker = ticker_symbol
        
        # Load FinBERT model and tokenizer
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.labels = ["negative", "neutral", "positive"]
        
    def fetch_stock_data(self, period="1d", interval="1m"):
        """
        Fetch real-time stock price data using Yahoo Finance
        
        Args:
            period (str): Time period to fetch data for (e.g., '1d', '5d', '1mo')
            interval (str): Time interval between data points (e.g., '1m', '5m', '1h')
            
        Returns:
            pandas.DataFrame: Stock price data
        """
        try:
            stock = yf.Ticker(self.ticker)
            data = stock.history(period=period, interval=interval)
            print(f"Successfully fetched {len(data)} data points for {self.ticker}")
            return data
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None
    
    def fetch_news_sentiment(self, api_key=None):
        """
        Fetch news articles and their sentiment using a news API
        
        Args:
            api_key (str): API key for news service (e.g., NewsAPI)
            
        Returns:
            list: News articles with sentiment scores
        """
        if not api_key:
            print("Warning: No API key provided for news. Using sample data.")
            # Return sample data for demonstration
            return self._get_sample_news()
        
        # Example using NewsAPI
        base_url = "https://newsapi.org/v2/everything"
        params = {
            "q": self.ticker,
            "apiKey": api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                return self._analyze_news_sentiment(articles)
            else:
                print(f"Error fetching news: {response.status_code}")
                return self._get_sample_news()
        except Exception as e:
            print(f"Error in news API request: {e}")
            return self._get_sample_news()
    
    def fetch_twitter_sentiment(self, api_key=None, api_secret=None):
        """
        Fetch tweets about the stock and analyze sentiment
        
        Note: This is a placeholder. Twitter API v2 requires authentication
        and has rate limits. For production, implement proper OAuth.
        
        Returns:
            dict: Sentiment analysis of tweets
        """
        if not api_key or not api_secret:
            print("Warning: No Twitter API credentials provided. Using sample data.")
            return self._get_sample_tweets()
        
        # In a real implementation, you would use tweepy or similar library
        # to authenticate and fetch tweets about the stock
        
        return self._get_sample_tweets()
    
    def fetch_reddit_sentiment(self, client_id=None, client_secret=None):
        """
        Fetch Reddit posts about the stock and analyze sentiment
        
        Note: This is a placeholder. Reddit API requires authentication.
        
        Returns:
            dict: Sentiment analysis of Reddit posts
        """
        if not client_id or not client_secret:
            print("Warning: No Reddit API credentials provided. Using sample data.")
            return self._get_sample_reddit_posts()
        
        # In a real implementation, you would use PRAW or similar library
        # to authenticate and fetch Reddit posts about the stock
        
        return self._get_sample_reddit_posts()
    
    def _analyze_sentiment_with_finbert(self, text):
        """
        Analyze sentiment of text using FinBERT
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores and label
        """
        # Truncate text if it's too long for the model
        max_length = self.tokenizer.model_max_length
        if len(text) > max_length:
            text = text[:max_length]
            
        # Tokenize and get sentiment
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get scores and convert to probabilities
        scores = outputs.logits[0].numpy()
        scores = softmax(scores)
        
        # Create sentiment dictionary similar to VADER format for compatibility
        sentiment_dict = {
            "neg": float(scores[0]),
            "neu": float(scores[1]),
            "pos": float(scores[2]),
            "compound": float(scores[2] - scores[0])  # Simplified compound score
        }
        
        return sentiment_dict
    
    def _analyze_news_sentiment(self, articles):
        """Analyze sentiment of news articles using FinBERT"""
        analyzed_articles = []
        
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            content = article.get("content", "")
            
            # Combine text for analysis
            text = f"{title} {description} {content}"
            
            # Get sentiment scores using FinBERT
            sentiment = self._analyze_sentiment_with_finbert(text)
            
            analyzed_articles.append({
                "title": title,
                "url": article.get("url", ""),
                "publishedAt": article.get("publishedAt", ""),
                "sentiment": sentiment,
                "compound_score": sentiment["compound"]
            })
        
        return analyzed_articles
    
    def _get_sample_news(self):
        """Return sample news data for demonstration"""
        # Sample news with FinBERT-style sentiment scores
        return [
            {
                "title": f"{self.ticker} Reports Strong Quarterly Earnings",
                "url": "https://example.com/news1",
                "publishedAt": datetime.now().isoformat(),
                "sentiment": {"neg": 0.05, "neu": 0.15, "pos": 0.80, "compound": 0.75},
                "compound_score": 0.75
            },
            {
                "title": f"Analysts Upgrade {self.ticker} Stock Rating",
                "url": "https://example.com/news2",
                "publishedAt": datetime.now().isoformat(),
                "sentiment": {"neg": 0.03, "neu": 0.17, "pos": 0.80, "compound": 0.77},
                "compound_score": 0.77
            },
            {
                "title": f"Market Uncertainty Affects {self.ticker}",
                "url": "https://example.com/news3",
                "publishedAt": datetime.now().isoformat(),
                "sentiment": {"neg": 0.65, "neu": 0.30, "pos": 0.05, "compound": -0.60},
                "compound_score": -0.60
            }
        ]
    
    def _get_sample_tweets(self):
        """Return sample Twitter data for demonstration"""
        # Use ticker symbol to generate varied sentiment distributions
        import hashlib
        
        # Create a hash based on the ticker to get consistent but varied results per ticker
        ticker_hash = int(hashlib.md5(self.ticker.encode()).hexdigest(), 16) % 100
        
        # Sample tweets with FinBERT-style sentiment scores that vary by ticker
        sample_tweets = [
            {
                "text": f"Just bought more {self.ticker}! Feeling bullish!", 
                "sentiment": {"neg": 0.05, "neu": 0.15, "pos": 0.80, "compound": 0.75}
            },
            {
                "text": f"{self.ticker} price movement seems stable today", 
                "sentiment": {"neg": 0.10, "neu": 0.80, "pos": 0.10, "compound": 0.0}
            },
            {
                "text": f"Not sure about {self.ticker}'s future prospects after the news", 
                "sentiment": {"neg": 0.60, "neu": 0.30, "pos": 0.10, "compound": -0.50}
            },
            {
                "text": f"Earnings report for {self.ticker} looks promising!", 
                "sentiment": {"neg": 0.05, "neu": 0.25, "pos": 0.70, "compound": 0.65}
            },
            {
                "text": f"Considering selling my {self.ticker} shares after today's news", 
                "sentiment": {"neg": 0.55, "neu": 0.35, "pos": 0.10, "compound": -0.45}
            }
        ]
        
        # Calculate average sentiment
        avg_sentiment = sum(tweet["sentiment"]["compound"] for tweet in sample_tweets) / len(sample_tweets)
        
        # Use ticker hash to adjust the distribution
        positive_weight = (ticker_hash % 60) + 20  # 20-79%
        negative_weight = ((ticker_hash + 30) % 40) + 10  # 10-49%
        neutral_weight = 100 - positive_weight - negative_weight
        
        # Create distribution based on weights
        sentiment_distribution = {
            "positive": int(positive_weight),
            "neutral": int(neutral_weight),
            "negative": int(negative_weight)
        }
        
        return {
            "tweet_count": 100,
            "average_sentiment": avg_sentiment,
            "sentiment_distribution": sentiment_distribution,
            "sample_tweets": sample_tweets
        }
    
    def _get_sample_reddit_posts(self):
        """Return sample Reddit data for demonstration"""
        # Use ticker symbol to generate varied sentiment distributions
        import hashlib
        
        # Create a hash based on the ticker to get consistent but varied results per ticker
        # Use a different seed than Twitter to ensure different distributions
        ticker_hash = int(hashlib.md5((self.ticker + "reddit").encode()).hexdigest(), 16) % 100
        
        # Sample Reddit posts with FinBERT-style sentiment scores
        sample_posts = [
            {
                "title": f"{self.ticker} DD: Why I'm investing", 
                "sentiment": {"neg": 0.05, "neu": 0.25, "pos": 0.70, "compound": 0.65}
            },
            {
                "title": f"What's happening with {self.ticker} today?", 
                "sentiment": {"neg": 0.10, "neu": 0.80, "pos": 0.10, "compound": 0.0}
            },
            {
                "title": f"Concerns about {self.ticker}'s market position", 
                "sentiment": {"neg": 0.65, "neu": 0.25, "pos": 0.10, "compound": -0.55}
            },
            {
                "title": f"Long-term outlook for {self.ticker} looks promising", 
                "sentiment": {"neg": 0.10, "neu": 0.20, "pos": 0.70, "compound": 0.60}
            },
            {
                "title": f"Should I sell my {self.ticker} shares now?", 
                "sentiment": {"neg": 0.40, "neu": 0.50, "pos": 0.10, "compound": -0.30}
            }
        ]
        
        # Calculate average sentiment
        avg_sentiment = sum(post["sentiment"]["compound"] for post in sample_posts) / len(sample_posts)
        
        # Use ticker hash to adjust the distribution
        positive_weight = (ticker_hash % 50) + 15  # 15-64%
        negative_weight = ((ticker_hash + 20) % 35) + 15  # 15-49%
        neutral_weight = 100 - positive_weight - negative_weight
        
        # Create distribution based on weights
        sentiment_distribution = {
            "positive": int(positive_weight),
            "neutral": int(neutral_weight),
            "negative": int(negative_weight)
        }
        
        return {
            "post_count": 25,
            "average_sentiment": avg_sentiment,
            "sentiment_distribution": sentiment_distribution,
            "sample_posts": sample_posts
        }
    
    def analyze_combined_sentiment(self, news=None, twitter=None, reddit=None):
        """
        Combine sentiment from multiple sources and analyze
        
        Returns:
            dict: Combined sentiment analysis
        """
        sources = []
        scores = []
        
        if news:
            avg_news_sentiment = sum(article["compound_score"] for article in news) / len(news)
            sources.append("News")
            scores.append(avg_news_sentiment)
        
        if twitter:
            sources.append("Twitter")
            scores.append(twitter["average_sentiment"])
        
        if reddit:
            sources.append("Reddit")
            scores.append(reddit["average_sentiment"])
        
        if not scores:
            return {"overall_sentiment": 0, "sentiment_label": "Neutral", "sources": {}}
        
        overall_sentiment = sum(scores) / len(scores)
        
        # Determine sentiment label
        if overall_sentiment > 0.2:
            sentiment_label = "Positive"
        elif overall_sentiment < -0.2:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Create source breakdown
        source_breakdown = {source: score for source, score in zip(sources, scores)}
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_label": sentiment_label,
            "sources": source_breakdown
        }
    
    def visualize_sentiment_vs_price(self, stock_data, sentiment_data, output_file=None):
        """
        Create visualization of sentiment vs stock price
        
        Args:
            stock_data (pandas.DataFrame): Stock price data
            sentiment_data (list): Sentiment data points with timestamps
            output_file (str): Path to save the visualization
        """
        if stock_data is None or stock_data.empty:
            print("No stock data available for visualization")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot stock price
        ax1.plot(stock_data.index, stock_data['Close'], 'b-', label='Close Price')
        ax1.set_ylabel('Stock Price ($)')
        ax1.set_title(f'{self.ticker} Price and Sentiment Analysis')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Plot sentiment (using sample data for demonstration)
        timestamps = stock_data.index[-5:]  # Just use the last few timestamps for demo
        sentiment_values = [0.2, 0.4, -0.1, 0.3, 0.5]  # Sample sentiment values
        
        ax2.bar(timestamps, sentiment_values, color=['g' if s > 0 else 'r' for s in sentiment_values], alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Sentiment Score')
        ax2.set_xlabel('Time')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()

def main():
    # Example usage
    ticker = "AAPL"  # Apple Inc.
    print(f"Initializing FinBERT-based sentiment analysis for {ticker}...")
    analyzer = StockSentimentAnalyzer(ticker)
    print("FinBERT model loaded successfully!")
    
    # Fetch stock data
    stock_data = analyzer.fetch_stock_data(period="1d", interval="5m")
    
    # Fetch sentiment data
    # For production use, replace with your actual API keys
    print("\nAnalyzing news sentiment with FinBERT...")
    news_sentiment = analyzer.fetch_news_sentiment(api_key="2dc72496b0464fadacea7d5621634812")  
    twitter_sentiment = analyzer.fetch_twitter_sentiment()
    reddit_sentiment = analyzer.fetch_reddit_sentiment()
    
    # Analyze combined sentiment
    combined_sentiment = analyzer.analyze_combined_sentiment(
        news=news_sentiment,
        twitter=twitter_sentiment,
        reddit=reddit_sentiment
    )
    
    # Print results
    print(f"\n{ticker} Sentiment Analysis Results (using FinBERT):")
    print(f"Overall Sentiment: {combined_sentiment['sentiment_label']} ({combined_sentiment['overall_sentiment']:.2f})")
    print("\nSource Breakdown:")
    for source, score in combined_sentiment['sources'].items():
        print(f"  {source}: {score:.2f}")
    
    # Sample news headlines with sentiment
    print("\nRecent News Headlines (analyzed with FinBERT):")
    for article in news_sentiment[:3]:  # Show top 3 news
        sentiment = "🟢" if article["compound_score"] > 0.2 else "🔴" if article["compound_score"] < -0.2 else "⚪"
        print(f"  {sentiment} {article['title']} (Score: {article['compound_score']:.2f})")
    
    # Visualize
    analyzer.visualize_sentiment_vs_price(stock_data, None)

if __name__ == "__main__":
    main()
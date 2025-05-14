# Stock Sentiment Analysis Tool

This tool performs sentiment analysis on stocks by gathering data from multiple sources and correlating it with real-time stock prices.

## Features

- Fetches real-time stock price data using Yahoo Finance
- Gathers sentiment data from:
  - Financial news articles
  - Twitter/X posts
  - Reddit discussions
- Analyzes sentiment using FinBERT (financial domain-specific BERT model)
- Visualizes correlation between sentiment and stock price movements
- Interactive web interface for real-time analysis

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download NLTK resources (first time only):

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
```

## Usage

### Interactive Web App

Run the Streamlit app for an interactive experience:

```bash
streamlit run app.py
```

This will launch a web interface where you can:
- Enter any stock ticker symbol
- Select time periods and intervals for analysis
- View real-time stock price charts
- See sentiment analysis from multiple sources
- Explore news headlines with sentiment scores
- Visualize sentiment distribution across social media

### Programmatic Usage

For programmatic use:

```python
from stock_sentiment_analysis import StockSentimentAnalyzer

# Initialize analyzer with a stock ticker
analyzer = StockSentimentAnalyzer("AAPL")

# Fetch stock data
stock_data = analyzer.fetch_stock_data(period="1d", interval="5m")

# Fetch sentiment data (replace with your API keys for production use)
news_sentiment = analyzer.fetch_news_sentiment(api_key="YOUR_NEWS_API_KEY")
twitter_sentiment = analyzer.fetch_twitter_sentiment(api_key="YOUR_TWITTER_API_KEY", api_secret="YOUR_TWITTER_API_SECRET")
reddit_sentiment = analyzer.fetch_reddit_sentiment(client_id="YOUR_REDDIT_CLIENT_ID", client_secret="YOUR_REDDIT_CLIENT_SECRET")

# Analyze combined sentiment
combined_sentiment = analyzer.analyze_combined_sentiment(
    news=news_sentiment,
    twitter=twitter_sentiment,
    reddit=reddit_sentiment
)

# Print results
print(f"Overall Sentiment: {combined_sentiment['sentiment_label']} ({combined_sentiment['overall_sentiment']:.2f})")

# Visualize
analyzer.visualize_sentiment_vs_price(stock_data, None)
```

## API Keys

To use the full functionality of this tool, you'll need API keys for:

1. **News API**: [NewsAPI](https://newsapi.org/) (Free tier available)
2. **Twitter API**: [Twitter Developer Platform](https://developer.twitter.com/en/docs/twitter-api) 
3. **Reddit API**: [Reddit API](https://www.reddit.com/dev/api/)

Note: The application will use sample data when API keys are not provided.

## Screenshots

The interactive app includes:
- Real-time stock price charts
- Overall sentiment analysis
- News sentiment with headlines
- Social media sentiment distribution
- Sample tweets and Reddit posts

## Extending the Tool

You can extend this tool by:

1. Adding more data sources (e.g., StockTwits, financial forums)
2. Implementing more sophisticated sentiment analysis algorithms
3. Setting up alerts for significant sentiment shifts
4. Adding user authentication for saving preferences
5. Implementing historical sentiment tracking and correlation analysis

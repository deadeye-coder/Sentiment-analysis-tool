"""
Stock Sentiment Analysis Interactive App

This Streamlit app provides an interactive interface for analyzing stock sentiment
using the StockSentimentAnalyzer class.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
from stock_sentiment_analysis import StockSentimentAnalyzer

# Set page configuration for a cleaner interface
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Stock Sentiment Analyzer powered by FinBERT"
    }
)

# Custom CSS with fullscreen button
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #FFC107;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #F44336;
        font-weight: bold;
    }
    /* Removed source-card styling */
    /* Social media item styling */
    .social-item {
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Fullscreen button */
    .fullscreen-button {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 99;
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        cursor: pointer;
    }
    .fullscreen-button:hover {
        background-color: #0D47A1;
    }
    
    /* Bare mode styles */
    .bare-mode .stApp {
        margin-top: -75px;
    }
    .bare-mode header {
        visibility: hidden;
    }
</style>

<script>
function toggleFullScreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
        document.body.classList.add('bare-mode');
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
            document.body.classList.remove('bare-mode');
        }
    }
}
</script>

<button onclick="toggleFullScreen()" class="fullscreen-button">Toggle Fullscreen</button>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>Stock Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("Analyze stock sentiment from news, Twitter, and Reddit using FinBERT")

# Initialize session state for caching results
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'news_sentiment' not in st.session_state:
    st.session_state.news_sentiment = None
if 'twitter_sentiment' not in st.session_state:
    st.session_state.twitter_sentiment = None
if 'reddit_sentiment' not in st.session_state:
    st.session_state.reddit_sentiment = None
if 'combined_sentiment' not in st.session_state:
    st.session_state.combined_sentiment = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None

# Sidebar for inputs
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)
    
    # Stock ticker input
    ticker = st.text_input("Stock Ticker Symbol", "AAPL").upper()
    
    # Time period selection
    period_options = {
        "1 Day": "1d",
        "5 Days": "5d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y"
    }
    period = st.selectbox("Time Period", list(period_options.keys()))
    
    # Interval selection
    interval_options = {
        "1 Minute": "1m",
        "5 Minutes": "5m",
        "15 Minutes": "15m",
        "30 Minutes": "30m",
        "1 Hour": "1h",
        "1 Day": "1d"
    }
    interval = st.selectbox("Interval", list(interval_options.keys()))
    
    # API Keys (optional)
    st.markdown("### API Keys (Optional)")
    st.markdown("Leave blank to use sample data")
    news_api_key = st.text_input("News API Key", "", type="password")
    
    # Advanced options in expander
    with st.expander("Advanced Options"):
        twitter_api_key = st.text_input("Twitter API Key", "", type="password")
        twitter_api_secret = st.text_input("Twitter API Secret", "", type="password")
        reddit_client_id = st.text_input("Reddit Client ID", "", type="password")
        reddit_client_secret = st.text_input("Reddit Client Secret", "", type="password")
    
    # Analysis button
    analyze_button = st.button("Analyze Sentiment")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (every 5 min)")
    
    # Display last update time
    if st.session_state.last_update_time:
        st.markdown(f"Last updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Function to run analysis
def run_analysis():
    with st.spinner("Initializing FinBERT model..."):
        # Create analyzer if it doesn't exist or ticker has changed
        if st.session_state.analyzer is None or st.session_state.analyzer.ticker != ticker:
            st.session_state.analyzer = StockSentimentAnalyzer(ticker)
    
    # Fetch stock data
    with st.spinner(f"Fetching stock data for {ticker}..."):
        st.session_state.stock_data = st.session_state.analyzer.fetch_stock_data(
            period=period_options[period],
            interval=interval_options[interval]
        )
    
    # Fetch sentiment data
    with st.spinner("Analyzing news sentiment..."):
        st.session_state.news_sentiment = st.session_state.analyzer.fetch_news_sentiment(
            api_key=news_api_key if news_api_key else None
        )
    
    with st.spinner("Analyzing social media sentiment..."):
        st.session_state.twitter_sentiment = st.session_state.analyzer.fetch_twitter_sentiment(
            api_key=twitter_api_key if twitter_api_key else None,
            api_secret=twitter_api_secret if twitter_api_secret else None
        )
        
        st.session_state.reddit_sentiment = st.session_state.analyzer.fetch_reddit_sentiment(
            client_id=reddit_client_id if reddit_client_id else None,
            client_secret=reddit_client_secret if reddit_client_secret else None
        )
    
    # Analyze combined sentiment
    st.session_state.combined_sentiment = st.session_state.analyzer.analyze_combined_sentiment(
        news=st.session_state.news_sentiment,
        twitter=st.session_state.twitter_sentiment,
        reddit=st.session_state.reddit_sentiment
    )
    
    # Update last update time
    st.session_state.last_update_time = datetime.now()

# Run analysis when button is clicked or auto-refresh is enabled
if analyze_button or (auto_refresh and (st.session_state.last_update_time is None or 
                                        datetime.now() - st.session_state.last_update_time > timedelta(minutes=5))):
    run_analysis()

# Display results if data is available
if st.session_state.combined_sentiment:
    # Create two columns for the main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stock price chart
        if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
            st.markdown("<h2 class='sub-header'>Stock Price</h2>", unsafe_allow_html=True)
            
            # Create interactive stock price chart with Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.stock_data.index,
                y=st.session_state.stock_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1E88E5', width=2)
            ))
            
            fig.update_layout(
                title=f"{ticker} Stock Price ({period_options[period]}, {interval_options[interval]})",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                hovermode="x unified",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # News sentiment
        st.markdown("<h2 class='sub-header'>Recent News</h2>", unsafe_allow_html=True)
        if st.session_state.news_sentiment:
            for article in st.session_state.news_sentiment[:5]:  # Show top 5 news
                sentiment_class = "sentiment-positive" if article["compound_score"] > 0.2 else \
                                "sentiment-negative" if article["compound_score"] < -0.2 else \
                                "sentiment-neutral"
                
                sentiment_emoji = "🟢" if article["compound_score"] > 0.2 else \
                                "🔴" if article["compound_score"] < -0.2 else "⚪"
                
                st.markdown(f"""
                <div style='margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 10px;'>
                    <span>{sentiment_emoji} <span class='{sentiment_class}'>
                    {article['compound_score']:.2f}</span></span>
                    <h4>{article['title']}</h4>
                    <a href="{article['url']}" target="_blank">Read more</a>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Overall sentiment
        st.markdown("<h2 class='sub-header'>Overall Sentiment</h2>", unsafe_allow_html=True)
        
        sentiment_label = st.session_state.combined_sentiment['sentiment_label']
        sentiment_score = st.session_state.combined_sentiment['overall_sentiment']
        
        sentiment_class = "sentiment-positive" if sentiment_label == "Positive" else \
                        "sentiment-negative" if sentiment_label == "Negative" else \
                        "sentiment-neutral"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px;'>
            <h1 class='{sentiment_class}'>{sentiment_label}</h1>
            <h2 class='{sentiment_class}'>{sentiment_score:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Source breakdown
        st.markdown("<h3>Source Breakdown</h3>", unsafe_allow_html=True)
        
        source_data = pd.DataFrame({
            'Source': list(st.session_state.combined_sentiment['sources'].keys()),
            'Score': list(st.session_state.combined_sentiment['sources'].values())
        })
        
        # Create a bar chart for source breakdown
        fig = px.bar(
            source_data, 
            x='Source', 
            y='Score',
            color='Score',
            color_continuous_scale=['#F44336', '#FFC107', '#4CAF50'],
            range_color=[-1, 1],
            height=300
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_range=[-1, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Social media sentiment distribution
        st.markdown("<h3>Social Media Sentiment</h3>", unsafe_allow_html=True)
        
        # Twitter sentiment
        if st.session_state.twitter_sentiment:
            st.markdown("<h4>Twitter</h4>", unsafe_allow_html=True)
            
            twitter_dist = st.session_state.twitter_sentiment['sentiment_distribution']
            twitter_df = pd.DataFrame({
                'Sentiment': list(twitter_dist.keys()),
                'Count': list(twitter_dist.values())
            })
            
            fig = px.pie(
                twitter_df,
                values='Count',
                names='Sentiment',
                color='Sentiment',
                color_discrete_map={
                    'positive': '#4CAF50',
                    'neutral': '#FFC107',
                    'negative': '#F44336'
                },
                height=250,
                hole=0.4,
                title=f"{ticker} Twitter Sentiment"
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                title_x=0.5,
                annotations=[dict(text=f"{st.session_state.twitter_sentiment['tweet_count']} tweets", x=0.5, y=0.5, font_size=12, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Reddit sentiment
        if st.session_state.reddit_sentiment:
            st.markdown("<h4>Reddit</h4>", unsafe_allow_html=True)
            
            reddit_dist = st.session_state.reddit_sentiment['sentiment_distribution']
            reddit_df = pd.DataFrame({
                'Sentiment': list(reddit_dist.keys()),
                'Count': list(reddit_dist.values())
            })
            
            fig = px.pie(
                reddit_df,
                values='Count',
                names='Sentiment',
                color='Sentiment',
                color_discrete_map={
                    'positive': '#4CAF50',
                    'neutral': '#FFC107',
                    'negative': '#F44336'
                },
                height=250,
                hole=0.4,
                title=f"{ticker} Reddit Sentiment"
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                title_x=0.5,
                annotations=[dict(text=f"{st.session_state.reddit_sentiment['post_count']} posts", x=0.5, y=0.5, font_size=12, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # No social media samples

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>Stock Sentiment Analyzer powered by FinBERT | Data from Yahoo Finance, News API, Twitter, and Reddit</p>
    <p>Note: When API keys are not provided, sample data is used for demonstration purposes.</p>
</div>
""", unsafe_allow_html=True)
"""
Stock Sentiment Analysis Kiosk Mode

This is a minimal version of the app designed for kiosk displays with no UI controls.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from stock_sentiment_analysis import StockSentimentAnalyzer

# Remove all Streamlit UI elements
st.set_page_config(
    page_title="Stock Sentiment Kiosk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide all Streamlit UI elements
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 0; padding-bottom: 0;}
    
    /* Custom styling for kiosk mode */
    body {
        background-color: #000000;
        color: white;
    }
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .ticker-symbol {
        font-size: 5rem;
        color: white;
        text-align: center;
        margin: 0;
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
</style>
""", unsafe_allow_html=True)

# Configuration - Change these values directly in the code for kiosk mode
TICKER = "AAPL"  # Stock ticker to display
AUTO_REFRESH_SECONDS = 300  # Refresh every 5 minutes
PERIOD = "1d"  # Time period
INTERVAL = "5m"  # Time interval

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = StockSentimentAnalyzer(TICKER)
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now() - timedelta(minutes=10)  # Force initial update
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'combined_sentiment' not in st.session_state:
    st.session_state.combined_sentiment = None

# Check if it's time to refresh
current_time = datetime.now()
if (current_time - st.session_state.last_update).total_seconds() > AUTO_REFRESH_SECONDS:
    with st.spinner(f"Updating data for {TICKER}..."):
        # Fetch stock data
        st.session_state.stock_data = st.session_state.analyzer.fetch_stock_data(
            period=PERIOD, interval=INTERVAL
        )
        
        # Fetch sentiment data (using sample data)
        news_sentiment = st.session_state.analyzer.fetch_news_sentiment()
        twitter_sentiment = st.session_state.analyzer.fetch_twitter_sentiment()
        reddit_sentiment = st.session_state.analyzer.fetch_reddit_sentiment()
        
        # Analyze combined sentiment
        st.session_state.combined_sentiment = st.session_state.analyzer.analyze_combined_sentiment(
            news=news_sentiment,
            twitter=twitter_sentiment,
            reddit=reddit_sentiment
        )
        
        # Update last update time
        st.session_state.last_update = current_time

# Display header with current time
st.markdown(f"<h1 class='main-header'>Stock Sentiment Kiosk</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 class='ticker-symbol'>{TICKER}</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

# Display data if available
if st.session_state.stock_data is not None and st.session_state.combined_sentiment is not None:
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Stock price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.stock_data.index,
            y=st.session_state.stock_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1E88E5', width=3)
        ))
        
        fig.update_layout(
            title=f"{TICKER} Stock Price ({PERIOD}, {INTERVAL})",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Overall sentiment
        sentiment_label = st.session_state.combined_sentiment['sentiment_label']
        sentiment_score = st.session_state.combined_sentiment['overall_sentiment']
        
        sentiment_class = "sentiment-positive" if sentiment_label == "Positive" else \
                        "sentiment-negative" if sentiment_label == "Negative" else \
                        "sentiment-neutral"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px;'>
            <h1 class='{sentiment_class}' style='font-size: 4rem;'>{sentiment_label}</h1>
            <h2 class='{sentiment_class}' style='font-size: 3rem;'>{sentiment_score:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Source breakdown
        source_data = pd.DataFrame({
            'Source': list(st.session_state.combined_sentiment['sources'].keys()),
            'Score': list(st.session_state.combined_sentiment['sources'].values())
        })
        
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
            yaxis_range=[-1, 1],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add some sample tweets/posts without boxes
        if 'news_sentiment' in st.session_state and st.session_state.news_sentiment:
            st.markdown("<h3 style='color: white;'>Latest Headlines</h3>", unsafe_allow_html=True)
            for article in st.session_state.news_sentiment[:2]:
                sentiment_class = "sentiment-positive" if article["compound_score"] > 0.2 else \
                                "sentiment-negative" if article["compound_score"] < -0.2 else \
                                "sentiment-neutral"
                
                sentiment_emoji = "🟢" if article["compound_score"] > 0.2 else \
                                "🔴" if article["compound_score"] < -0.2 else "⚪"
                
                st.markdown(f"""
                <div style='margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px;'>
                    <p style='color: white;'><b>{sentiment_emoji} {article['title']}</b></p>
                </div>
                """, unsafe_allow_html=True)

# Auto-refresh the page
st.markdown(f"""
<script>
    // Refresh the page every {AUTO_REFRESH_SECONDS} seconds
    setTimeout(function(){{
        window.location.reload();
    }}, {AUTO_REFRESH_SECONDS * 1000});
</script>
""", unsafe_allow_html=True)
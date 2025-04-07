import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Import helper functions from utils.py
from utils import (
    compute_engagement,
    get_engagement_string,
    compute_keyword_engagement,
    create_persona_tweet
)
# Import GenAI from your existing genai.py file
from genai import GenAI

# Optional: Load environment variables from a .env file (ensure .env is in .gitignore)
from dotenv import load_dotenv
load_dotenv()

def main():
    # Set page configuration and add a futuristic cyberpunk theme
    st.set_page_config(page_title="CyberTweet Analyzer", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        body {
            background-color: #0f0f0f;
            color: #f8f8f2;
            font-family: 'Orbitron', sans-serif;
        }
        .stButton>button {
            background-color: #ff79c6;
            color: #0f0f0f;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #bd93f9;
        }
        .neon-text {
            color: #50fa7b;
            text-shadow: 0 0 5px #50fa7b, 0 0 10px #50fa7b, 0 0 20px #ff79c6, 0 0 30px #ff79c6, 0 0 40px #ff79c6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Homepage", "Keyword Engagement", "Persona Tweet"])
    
    # Load the OpenAI API key from an environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return
    genai = GenAI(openai_api_key)
    
    # Homepage: Upload CSV and analyze tweet engagement
    if page == "Homepage":
        st.markdown('<h1 class="neon-text">CyberTweet Analyzer</h1>', unsafe_allow_html=True)
        st.header("Upload your tweet CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            # Ensure the CSV contains the required columns: tweet, favorite_count, and view_count
            if "favorite_count" not in df.columns or "view_count" not in df.columns or "tweet" not in df.columns:
                st.error("CSV must include 'tweet', 'favorite_count', and 'view_count' columns")
                return
            with st.spinner("Analyzing engagement ..."):
                df = compute_engagement(df)
                engagement_analysis = get_engagement_string(df, genai)
            st.subheader("Top Tweets")
            df_sorted = df.sort_values(by="engagement", ascending=False)[["tweet", "engagement"]]
            st.dataframe(df_sorted)
            st.markdown(
                f"<div style='padding: 10px; border: 1px solid #ff79c6; border-radius: 5px;'>{engagement_analysis}</div>",
                unsafe_allow_html=True
            )
            # Save the dataframe and analysis to session state for use on other pages
            st.session_state.df = df
            st.session_state.engagement_analysis = engagement_analysis

    # Keyword Engagement: Analyze keywords' impact on engagement
    elif page == "Keyword Engagement":
        st.markdown('<h1 class="neon-text">CyberTweet Analyzer</h1>', unsafe_allow_html=True)
        st.header("Keyword Engagement")
        if "df" not in st.session_state:
            st.warning("Please upload a CSV file on the Homepage first.")
        else:
            keywords_string = st.text_input("Keywords (comma-separated). You can input multiple keywords if they are separated by commas.")
            if st.button("Analyze"):
                df = st.session_state.df
                df_keywords = compute_keyword_engagement(df, keywords_string)
                # Create a bar plot using Plotly Express
                fig = px.bar(
                    df_keywords,
                    x="keyword",
                    y=["engagement_true", "engagement_false"],
                    barmode="group",
                    title="Keyword Engagement"
                )
                # Add BH corrected p-value above the "true" bar for each keyword
                for i, row in df_keywords.iterrows():
                    fig.add_annotation(
                        x=row["keyword"],
                        y=row["engagement_true"],
                        text=f"p={row['pvalue_bh']:.3f}",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="#ff79c6")
                    )
                st.plotly_chart(fig, use_container_width=True)
    
    # Persona Tweet: Generate a new tweet in the user's style
    elif page == "Persona Tweet":
        st.markdown('<h1 class="neon-text">CyberTweet Analyzer</h1>', unsafe_allow_html=True)
        st.header("Persona Tweet")
        if "df" not in st.session_state or "engagement_analysis" not in st.session_state:
            st.warning("Please complete the analysis on the Homepage first.")
        else:
            topic = st.text_input("Topic (text or URL)")
            if st.button("Create tweet"):
                with st.spinner("Creating tweet..."):
                    tweet_html = create_persona_tweet(topic, st.session_state.df, st.session_state.engagement_analysis, genai)
                st.markdown(tweet_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

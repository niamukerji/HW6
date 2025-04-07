import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

def compute_engagement(df):
    """
    Compute engagement as favorite_count divided by view_count.
    Avoid division by zero.
    """
    df["engagement"] = df.apply(
        lambda row: row["favorite_count"] / row["view_count"] if row["view_count"] != 0 else 0,
        axis=1
    )
    return df

def get_engagement_string(df, genai):
    """
    Use AI (via GenAI) to analyze the tweet texts and their engagement metrics.
    Uses the top 10 tweets by engagement as context.
    """
    # Get top 10 tweets by engagement
    sample = df.sort_values(by="engagement", ascending=False).head(10)
    tweets_info = "\n".join(
        [f"Tweet: {row['tweet']}, Engagement: {row['engagement']:.2f}" for index, row in sample.iterrows()]
    )
    prompt = (
        f"Analyze the following tweets and their engagement metrics. "
        f"Provide insights on what makes these tweets engaging:\n{tweets_info}"
    )
    response = genai.generate_text(
        prompt,
        instructions="You are an AI analyst specializing in social media engagement. "
                     "Analyze the tweet texts and their engagement metrics and provide clear insights."
    )
    return response

def compute_keyword_engagement(df, keywords_string):
    """
    For each comma-separated keyword, compute:
      - Mean engagement for tweets that contain the keyword and those that do not.
      - p-value from a t-test comparing the two groups.
      - Apply Benjamini-Hochberg correction for multiple testing.
    Returns a DataFrame with columns: keyword, pvalue_bh, engagement_false, engagement_true.
    """
    keywords = [kw.strip() for kw in keywords_string.split(",") if kw.strip()]
    results = []
    for kw in keywords:
        mask = df["tweet"].str.contains(kw, case=False, na=False)
        engagement_true = df[mask]["engagement"].mean() if mask.sum() > 0 else 0
        engagement_false = df[~mask]["engagement"].mean() if (~mask).sum() > 0 else 0
        # Use t-test if there are enough samples in each group; otherwise default p-value to 1.0
        if mask.sum() > 1 and (~mask).sum() > 1:
            _, pvalue = ttest_ind(
                df[mask]["engagement"],
                df[~mask]["engagement"],
                equal_var=False,
                nan_policy='omit'
            )
        else:
            pvalue = 1.0
        results.append({
            "keyword": kw,
            "engagement_true": engagement_true,
            "engagement_false": engagement_false,
            "pvalue": pvalue
        })
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values("pvalue")
        n = len(df_results)
        df_results["rank"] = np.arange(1, n + 1)
        df_results["pvalue_bh"] = df_results["pvalue"] * n / df_results["rank"]
        df_results["pvalue_bh"] = df_results["pvalue_bh"].apply(lambda x: min(x, 1.0))
        df_results = df_results.sort_values("keyword")
        df_results = df_results[["keyword", "pvalue_bh", "engagement_false", "engagement_true"]]
    return df_results

def create_persona_tweet(topic, df, engagement_analysis_string, genai):
    """
    Use AI to generate a new tweet that mimics the user's style.
    The function takes a topic (or URL) and uses a sample of existing tweets plus
    engagement insights to create an engaging tweet.
    """
    # If the topic appears to be a URL, you might fetch its text. Here we use a placeholder.
    if topic.lower().startswith("http"):
        topic_content = f"Content from {topic}"
    else:
        topic_content = topic

    # Sample a few tweets to provide context for the user's style
    sample = df.sample(min(5, len(df)))
    tweets_sample = "\n".join([row["tweet"] for _, row in sample.iterrows()])
    
    prompt = (
        f"Based on the following tweet samples and engagement insights, create a new, engaging tweet on the topic: {topic_content}.\n"
        f"Tweet samples:\n{tweets_sample}\n\n"
        f"Engagement insights:\n{engagement_analysis_string}\n\n"
        f"Ensure the tweet mimics the user's style and has a futuristic, cyberpunk vibe."
    )
    
    tweet_text = genai.generate_text(
        prompt,
        instructions="You are an AI that mimics a user's tweet style and creates engaging tweets in a cyberpunk tone."
    )
    # Format the generated tweet using GenAI's display_tweet function so it looks like a real tweet
    tweet_html = genai.display_tweet(text=tweet_text, screen_name="AI Persona")
    return tweet_html

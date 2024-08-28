# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:40:47 2024

@author: lukas.knoll
"""

import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import os

nltk.download("stopwords")
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('punkt')

#Defining working directory (@Prof. Manshauen & Herr Dietrich Needs to be changed accordingly)
os.chdir(r'C:\Users\Lukas\OneDrive - Provadis Hochschule\Semester 4\Masterthesis\Hauptforschungen\Research_BTC_SA\BTC_SentimentAnalysis')

csv_path = os.path.join('Datengrundlage', 'BTC_SentimentAnalysis_Masters.csv')
btc_data = pd.read_csv(csv_path)

#Deleting non text or date + duplicates
btc_data.dropna(axis=0, subset=["date", "text"], inplace=True)
btc_data.drop_duplicates(inplace=True)

#Converting data types
btc_data = btc_data.convert_dtypes()
btc_data["date"] = pd.to_datetime(btc_data["date"], errors="coerce")
btc_data.dropna(inplace=True)
btc_data["date"] = pd.to_datetime(btc_data["date"], format="%Y-%m-%d %H:%M:%S")

#Creating new column 'Follower size'
btc_data['Follower_Size'] = 'Other'
btc_data['user_followers'] = pd.to_numeric(btc_data['user_followers'], errors='coerce').astype('Int64')
btc_data['user_favourites'] = pd.to_numeric(btc_data['user_favourites'], errors='coerce').astype('Int64')

#Setting values based on conditions
btc_data.loc[(btc_data['user_followers'] >= 10) & (btc_data['user_followers'] < 100), 'Follower_Size'] = '10 - 99 followers'
btc_data.loc[(btc_data['user_followers'] >= 100) & (btc_data['user_followers'] < 1000), 'Follower_Size'] = '100 - 999 followers'
btc_data.loc[(btc_data['user_followers'] >= 1000) & (btc_data['user_followers'] < 10000), 'Follower_Size'] = '1000 - 9999 followers'
btc_data.loc[(btc_data['user_followers'] >= 10000) & (btc_data['user_followers'] < 100000), 'Follower_Size'] = '10000 - 99999 followers'
btc_data.loc[(btc_data['user_followers'] >= 100000), 'Follower_Size'] = '100000 or more followers'

#Function to preprocess a tweet into words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def tweet_to_words(tweet):
    tweet = tweet.lower()
    words = word_tokenize(tweet)
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(token) for token in words]
    return words

#Cleaning the text
btc_data['clean_text'] = btc_data['text'].apply(lambda x: ' '.join(tweet_to_words(x)))

#Sentiment analysis
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(tweet):
    sentiment_dict = analyzer.polarity_scores(tweet)
    return sentiment_dict

btc_data['Vader_Comp'] = btc_data['clean_text'].apply(lambda x: analyze_sentiment(x)['compound'])

#Filtering for the worst sentiment scores with threshold -0.1
threshold = -0.1
worst_btc_data_negative = btc_data[btc_data['Vader_Comp'] < threshold]

# Define topics and keywords
topics = {
    'Regulation': ['regulation', 'regulatory', 'compliance', 'laws', 'rules'],
    'Security': ['security', 'hack', 'fraud', 'cybersecurity', 'breach', 'scam'],
    'Finance': ['financial', 'investment', 'market', 'economy', 'trading'],
    'Technology': ['technology', 'innovation', 'software', 'hardware', 'AI'],
    'Volatility': ['volatility', 'volatile', 'price swings', 'market swings', 'price fluctuations'],
    'Economic Policy': ['economic policy', 'fiscal policy', 'monetary policy', 'central bank', 'interest rates', 'policy', 'economic forecast', 'subsidies', 'tax', 'inflation'],
    'Fiat Currencies': ['fiat', 'USD', 'EUR', 'GBP', 'currency exchange', 'fiat money']
}

#Categorizing btc_data by topic
def categorize_tweet(tweet, topics):
    for topic, keywords in topics.items():
        for keyword in keywords:
            if keyword in tweet.lower():
                return topic
    return 'Other'

worst_btc_data_negative['Topic'] = worst_btc_data_negative['clean_text'].apply(lambda x: categorize_tweet(x, topics))

#Evaluating topics in the worst sentiment btc_data
topic_counts_negative = worst_btc_data_negative['Topic'].value_counts()

#Printing the topic counts
print("Topic distribution in worst sentiment btc_data (threshold < -0.1):")
print(topic_counts_negative)

#Plotting the distribution of the worst sentiment btc_data
plt.figure(figsize=(12, 6))
sns.histplot(worst_btc_data_negative['Vader_Comp'], bins=30, kde=True)
plt.title('Distribution of Worst Compound Sentiment Scores (threshold < -0.1)')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Frequency')
plt.show()

topic_order = ['Regulation', 'Security', 'Finance', 'Technology', 'Volatility', 'Economic Policy', 'Fiat Currencies', 'Other']  # Ensure the order is consistent

#Defining colors for each topic
topic_colors = {
    'Regulation': 'blue',
    'Security': 'green',
    'Finance': 'red',
    'Technology': 'purple',
    'Volatility': 'orange',
    'Economic Policy': 'yellow',
    'Fiat Currencies': 'pink',
    'Other': 'grey'
}

palette = [topic_colors[topic] for topic in topic_order]

#Plotting the worst sentiment btc_data by Topic for threshold < -0.1
plt.figure(figsize=(12, 6))
sns.boxplot(x='Topic', y='Vader_Comp', data=worst_btc_data_negative, order=topic_order, palette=palette)
plt.title('Worst Sentiment Scores by Topic (threshold =< -0.1)')
plt.xlabel('Topic')
plt.ylabel('Compound Sentiment Score')
plt.xticks(rotation=45)
plt.show()

#Calculating and printing statistics for the worst sentiment scores by Topic
topics_negative = worst_btc_data_negative['Topic'].unique()
topic_stats_negative = {}

def analyze_sentiment_Comp_stats(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    median_score = np.median(scores)
    quartiles = np.percentile(scores, [25, 50, 75])
    mean_score = np.mean(scores)
    
    return {
        'min': min_score,
        'max': max_score,
        'median': median_score,
        'quartiles': quartiles,
        'mean': mean_score
    }

for topic in topics_negative:
    topic_btc_data = worst_btc_data_negative[worst_btc_data_negative['Topic'] == topic]
    compound_scores = topic_btc_data['Vader_Comp']
    stats = analyze_sentiment_Comp_stats(compound_scores)
    topic_stats_negative[topic] = stats

#Output the statistical values
for topic, stats in topic_stats_negative.items():
    print(f"Topic: {topic}")
    print(f"Min: {stats['min']:.2f}")
    print(f"Max: {stats['max']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"Quartiles: {stats['quartiles']}")
    print(f"Mean: {stats['mean']:.2f}")
    print()

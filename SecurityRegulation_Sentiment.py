# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:02:19 2024

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
btc_data.loc[(btc_data['user_followers'] >= 0) & (btc_data['user_followers'] < 100), 'Follower_Size'] = '10 - 99 followers'
btc_data.loc[(btc_data['user_followers'] >= 100) & (btc_data['user_followers'] < 1000), 'Follower_Size'] = '100 - 999 followers'
btc_data.loc[(btc_data['user_followers'] >= 1000) & (btc_data['user_followers'] < 10000), 'Follower_Size'] = '1000 - 9999 followers'
btc_data.loc[(btc_data['user_followers'] >= 10000) & (btc_data['user_followers'] < 100000), 'Follower_Size'] = '10000 - 99999 followers'
btc_data.loc[(btc_data['user_followers'] >= 100000), 'Follower_Size'] = '100000 or more followers'

#Defining keywords for regulation and security
regulation_keywords = [
    "regulation", "regulatory", "compliance", "laws", "rules", "legislation", "legal", "policy", "enforcement", 
    "governance", "sanctions", "AML", "KYC", "anti-money laundering", "know your customer", "transparency", 
    "reporting", "due diligence", "financial crime", "regulatory framework", "legal framework", "jurisdiction", 
    "oversight", "prosecution", "litigation", "penalties", "fines", "legal action", "regulatory compliance", 
    "blockchain regulation", "ICO regulation", "token regulation", "crypto regulation", "crypto laws", 
    "crypto policies", "crypto compliance", "regulatory risk", "financial regulation", "market regulation", 
    "trade regulation", "international regulation", "regulatory landscape", "regulatory environment", 
    "regulatory challenges", "regulatory hurdles", "regulatory barriers", "legal issues", "legal risks", 
    "legal challenges", "regulatory updates", "regulatory changes", "regulatory developments", "legal updates", 
    "legal developments"
]

security_keywords = [
    "security", "hack", "cybersecurity", "data breach", "breach", "privacy", "protection", "scam", "scams", 
    "theft", "insurance", "auditing", "audit", "supervision", "supervisory", "risk management", "security protocol", 
    "security measures", "security breach", "vulnerability", "penetration testing", "incident response", "threat", 
    "malware", "phishing", "ransomware", "exploit", "attack vector", "zero-day", "encryption", "secure", 
    "tokenization", "security risks", "security updates", "security changes", "security developments"
]

#Filtering btc_data based on topics 'Regulation' and 'Security'
regulation_pattern = '|'.join(regulation_keywords)
security_pattern = '|'.join(security_keywords)

regulation_btc_data = btc_data[btc_data['text'].str.contains(regulation_pattern, case=False, na=False)]
security_btc_data = btc_data[btc_data['text'].str.contains(security_pattern, case=False, na=False)]

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
regulation_btc_data['clean_text'] = regulation_btc_data['text'].apply(lambda x: ' '.join(tweet_to_words(x)))
security_btc_data['clean_text'] = security_btc_data['text'].apply(lambda x: ' '.join(tweet_to_words(x)))

#Sentiment analysis
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(tweet):
    sentiment_dict = analyzer.polarity_scores(tweet)
    return sentiment_dict

regulation_btc_data['Vader_Comp'] = regulation_btc_data['clean_text'].apply(lambda x: analyze_sentiment(x)['compound'])
security_btc_data['Vader_Comp'] = security_btc_data['clean_text'].apply(lambda x: analyze_sentiment(x)['compound'])

#Filtering for the worst sentiment scores
threshold = -0.1
worst_regulation_btc_data = regulation_btc_data[regulation_btc_data['Vader_Comp'] < threshold]
worst_security_btc_data = security_btc_data[security_btc_data['Vader_Comp'] < threshold]

#Evaluating topics in the worst sentiment btc_data
def get_top_topics(btc_data, num_topics=10):
    all_words = ' '.join(btc_data['clean_text']).split()
    word_freq = Counter(all_words)
    return word_freq.most_common(num_topics)

worst_regulation_topics = get_top_topics(worst_regulation_btc_data)
worst_security_topics = get_top_topics(worst_security_btc_data)

#Printing the top topics
print("Top topics in worst sentiment regulation btc_data:")
for topic, freq in worst_regulation_topics:
    print(f"{topic}: {freq}")

print("\nTop topics in worst sentiment security btc_data:")
for topic, freq in worst_security_topics:
    print(f"{topic}: {freq}")

#Plotting the worst sentiment btc_data for regulation
plt.figure(figsize=(12, 6))
sns.histplot(worst_regulation_btc_data['Vader_Comp'], bins=30, kde=True)
plt.title('Distribution of Worst Compound Sentiment Scores for Regulation btc_data')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Frequency')
plt.show()

#Plotting the worst sentiment btc_data for security
plt.figure(figsize=(12, 6))
sns.histplot(worst_security_btc_data['Vader_Comp'], bins=30, kde=True)
plt.title('Distribution of Worst Compound Sentiment Scores for Security btc_data')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Frequency')
plt.show()

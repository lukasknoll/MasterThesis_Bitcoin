# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:29:25 2024

@author: lukas.knoll
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import nltk
import math
import datetime as dt
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
nltk.download("stopwords")
from nltk.corpus import stopwords

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = [10, 10]
from textblob import TextBlob
nltk.download('vader_lexicon')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_rows', None)


#Defining working directory (@Prof. Manshauen & Herr Dietrich Needs to be changed accordingly)
os.chdir(r'C:\Users\Lukas\OneDrive - Provadis Hochschule\Semester 4\Masterthesis\Hauptforschungen\Research_BTC_SA\BTC_SentimentAnalysis')

csv_path = os.path.join('Datengrundlage', 'BTC_SentimentAnalysis_Masters.csv')
btc_data = pd.read_csv(csv_path)


mean_vader_comp = btc_data['Vader_Comp'].mean()

print(f"The mean of column 'Vader_Comp' is: {mean_vader_comp}")

min_date = btc_data['date'].min()
max_date = btc_data['date'].max()

#timeframe of dataframe

print("Minimum Date:", min_date)
print("Maximum Date:", max_date)

#timeseries

btc_data['date_clean'] = pd.to_datetime(btc_data['date_clean'])

#Calculation of daily average compound score
daily_avg_compound = btc_data.groupby(btc_data['date_clean'].dt.date)['Vader_Comp'].mean()

#Calculation of sentiment score median
daily_median_compound = btc_data.groupby(btc_data['date_clean'].dt.date)['Vader_Comp'].median()

#plotting with median
plt.figure(figsize=(10, 6))
plt.plot(daily_median_compound.index, daily_median_compound.values, color='blue', label='Daily Average Compound Score')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of BTC Tweets Over Time (Daily Median)')
plt.legend()
plt.grid(True)
plt.show()

#correlation analysis
import numpy as np
import requests

#Fetching historical Bitcoin price data from Binance API
def fetch_binance_data(api_key, start_time, end_time, symbol='BTCUSDT', interval='1d', limit=1000):
    all_data = []
    start_time = pd.Timestamp(start_time).timestamp() * 1000
    end_time = pd.Timestamp(end_time).timestamp() * 1000
    while True:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={int(start_time)}&endTime={int(end_time)}&limit={limit}"
        headers = {
            'X-MBX-APIKEY': api_key
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][0] + 1  # Update start time for the next request
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)
    return df[['close']]

api_key = 'BINANCE-KEY'

start_time = '2021-02-05'
end_time = '2023-03-05'

bitcoin_data = fetch_binance_data(api_key, start_time, end_time)

merged_data_avg = bitcoin_data.merge(daily_avg_compound, how='left', left_index=True, right_index=True)

#Interpolation of missing data using nearest neighbor
filtered_data_avg = merged_data_avg.dropna(subset=['Vader_Comp'])

merged_data_avg['Vader_Comp_filled'] = merged_data_avg['Vader_Comp'].interpolate(method='nearest')

plt.figure(figsize=(10, 6))
merged_data_avg['Vader_Comp_filled'].plot(color='blue', label='Daily Average Compound Score')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of BTC Tweets Over Time (Daily Average)')
plt.legend()
plt.grid(True)
plt.show()

#Calculation of moving average (30)
window_size = 30  # You can adjust this as needed
moving_avg = merged_data_avg['Vader_Comp_filled'].rolling(window=window_size).mean()

#Plotting the original sentiment scores and moving average
plt.figure(figsize=(10, 6))
merged_data_avg['Vader_Comp_filled'].plot(color='blue', label='Daily Average Compound Score')
moving_avg.plot(color='red', label='Moving Average ({})'.format(window_size))
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of BTC Tweets Over Time (Daily Average)')
plt.legend()
plt.grid(True)
plt.show()

#Plotting the Bitcoin closing chart
#Calculation of moving average for close price
window_size = 30  # Beispiel für ein 30-Tage-Fenster
moving_avg = bitcoin_data['close'].rolling(window=window_size).mean()

#Actual plotting
plt.figure(figsize=(10, 6))
bitcoin_data['close'].plot(color='blue', label='Daily Closing Price')
moving_avg.plot(color='red', label='Moving Average ({})'.format(window_size))
plt.xlabel('Date')
plt.ylabel('Bitcoin Closing Price (USD)')
plt.title('Bitcoin Closing Price Over Time')
plt.legend()
plt.grid(True)
plt.show()


#Calculation of Exponential Moving Average (30)
span = 30  # You can adjust this as needed
ema = merged_data_avg['Vader_Comp_filled'].ewm(span=span, adjust=False).mean()

# Plotting the original sentiment scores and the EMA
plt.figure(figsize=(10, 6))
merged_data_avg['Vader_Comp_filled'].plot(color='blue', label='Daily Average Compound Score')
ema.plot(color='red', label='EMA (Span = {})'.format(span))
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of BTC Tweets Over Time (Daily Average)')
plt.legend()
plt.grid(True)
plt.show()

correlation_avg = filtered_data_avg['close'].corr(filtered_data_avg['Vader_Comp'])

print("Correlation between Bitcoin close price and average sentiment score:", correlation_avg)

plt.figure(figsize=(10, 6))
plt.scatter(filtered_data_avg['Vader_Comp'], filtered_data_avg['close'], alpha=0.5)
plt.title('Correlation between Bitcoin Close Price and Average Sentiment Score')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Bitcoin Close Price (USD)')
plt.grid(True)
plt.show()

#Calculation using the correlation coefficient
correlation_coefficient = filtered_data_avg['Vader_Comp'].corr(filtered_data_avg['close'])

#Plotting with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='Vader_Comp', y='close', data=filtered_data_avg, scatter_kws={'alpha':0.5})
plt.title('Correlation between Bitcoin Close Price and Average Sentiment Score')
plt.xlabel('Median Sentiment Score')
plt.ylabel('Bitcoin Close Price (USD)')
plt.grid(True)
plt.text(0.1, 0.9, f'Correlation: {correlation_coefficient:.2f}', ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

#Calcualtion of median
daily_median_compound = btc_data.groupby(btc_data['date_clean'].dt.date)['Vader_Comp'].median()

merged_data_median = bitcoin_data.merge(daily_median_compound, how='left', left_index=True, right_index=True)

merged_data_median['Vader_Comp_filled'] = merged_data_median['Vader_Comp'].interpolate(method='nearest')
filtered_data_median = merged_data_median.dropna(subset=['Vader_Comp'])


#Plotting sentiment data over time
plt.figure(figsize=(10, 6))
merged_data_median['Vader_Comp_filled'].plot(color='blue', label='Daily Median Compound Score')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of BTC Tweets Over Time (Daily Median)')
plt.legend()
plt.grid(True)
plt.show()


correlation_median = filtered_data_median['close'].corr(filtered_data_median['Vader_Comp'])

plt.figure(figsize=(10, 6))
plt.scatter(filtered_data_median['Vader_Comp'], filtered_data_median['close'], alpha=0.5)
plt.title('Correlation between Bitcoin Close Price and Median Sentiment Score')
plt.xlabel('Median Sentiment Score')
plt.ylabel('Bitcoin Close Price (USD)')
plt.grid(True)
plt.show()

import seaborn as sns

#Plotting with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='Vader_Comp', y='close', data=filtered_data_median, scatter_kws={'alpha':0.5})
plt.title('Correlation between Bitcoin Close Price and Median Sentiment Score')
plt.xlabel('Median Sentiment Score')
plt.ylabel('Bitcoin Close Price (USD)')
plt.grid(True)
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

#Combining all words into a single string
all_words = ' '.join(btc_data['hashtags'])

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    sentiment_score = analyzer.polarity_scores(word)['compound']
    if sentiment_score < -0.1:
        return 'red'
    elif sentiment_score > 0:
        return 'green'
    else:
        return 'grey'

#Creation of WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=1000).generate(all_words)

#Plotting WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axis
plt.show()


#Creation of histogram
plt.hist(merged_data_avg['Vader_Comp_filled'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Average Compound Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Daily Average Compound Sentiment Scores')
plt.grid(True)
plt.show()

min_value = merged_data_avg['Vader_Comp_filled'].min()
max_value = merged_data_avg['Vader_Comp_filled'].max()

print("Min:", min_value)
print("Max:", max_value)

import numpy as np
import matplotlib.pyplot as plt

counts, bins, _ = plt.hist(merged_data_avg['Vader_Comp_filled'], bins=10)

bin_centers = 0.5 * (bins[:-1] + bins[1:])

#Printing frequency of sentiment scores
for score, count in zip(bin_centers, counts):
    print(f"Sentiment Score: {score:.2f}, Frequency: {int(count)}")

#Calculation of the overall average for the column 'Vader_Comp_filled'
overall_avg_vader_comp = merged_data_avg['Vader_Comp_filled'].mean()

#Printing the overall average
print("Overall Average Vader_Comp_filled:", overall_avg_vader_comp)

#Plotting the histogram of median sentiment scores
plt.hist(merged_data_median['Vader_Comp_filled'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Median Compound Sentiment Score')
plt.ylabel('Frequency')
plt.title('Distribution of Daily Median Compound Sentiment Scores')
plt.grid(True)
plt.show()

min_value = merged_data_median['Vader_Comp_filled'].min()
max_value = merged_data_median['Vader_Comp_filled'].max()

print("Min:", min_value)
print("Max:", max_value)

counts, bins, _ = plt.hist(merged_data_median['Vader_Comp_filled'], bins=10)

bin_centers = 0.5 * (bins[:-1] + bins[1:])

#Calculation of the overall average for the column 'Vader_Comp_filled'
for score, count in zip(bin_centers, counts):
    print(f"Sentiment Score: {score:.2f}, Frequency: {int(count)}")

#Calculation of the overall median for the column 'Vader_Comp_filled'
overall_median_vader_comp = merged_data_median['Vader_Comp_filled'].mean()

#Printing the overall median
print("Overall median Vader_Comp_filled:", overall_median_vader_comp)

#Calculation of quartiles for column 'Vader_Comp_filled'
first_quartile = merged_data_avg['Vader_Comp_filled'].quantile(0.25)
second_quartile = merged_data_avg['Vader_Comp_filled'].quantile(0.5)  # Median
third_quartile = merged_data_avg['Vader_Comp_filled'].quantile(0.75)

#Printing calculated quartiles
print("Erstes Quartil (25. Perzentil):", first_quartile)
print("Zweites Quartil (Median, 50. Perzentil):", second_quartile)
print("Drittes Quartil (75. Perzentil):", third_quartile)



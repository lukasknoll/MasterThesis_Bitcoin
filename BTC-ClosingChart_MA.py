# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:34:32 2024

@author: lukas.knoll
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

#Defining working directory (@Prof. Manshauen & Herr Dietrich Needs to be changed accordingly)
os.chdir(r'C:\Users\Lukas\OneDrive - Provadis Hochschule\Semester 4\Masterthesis\Hauptforschungen\Research_BTC_SA\BTC_SentimentAnalysis')

csv_path = os.path.join('Datengrundlage', 'BTC-USD.csv')
btcusd_data = pd.read_csv(csv_path)

start_date = '2021-02-01'
end_date = '2023-03-31'
filtered_data = btcusd_data[start_date:end_date]

window_size = 30
filtered_data['MA_30'] = filtered_data['Close'].rolling(window=window_size).mean()

plt.figure(figsize=(10, 6))
filtered_data['Close'].plot(color='blue', label='BTC Closing Price')
filtered_data['MA_30'].plot(color='red', label='Moving Average (30)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Bitcoin Closing Price and 30-Day Moving Average (Feb 2021 - Mar 2023)')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#Defining working directory (@Prof. Manshauen & Herr Dietrich Needs to be changed accordingly)
os.chdir(r'C:\Users\Lukas\OneDrive - Provadis Hochschule\Semester 4\Masterthesis\Hauptforschungen\Research_BTC_SA\BTC_SentimentAnalysis')

base_path = os.path.join('Datengrundlage', 'GBPUSD=X.csv')
gbp_usd = pd.read_csv(base_path, parse_dates=['Date'])

base_path = os.path.join('Datengrundlage', 'EURUSD=X.csv')
eur_usd = pd.read_csv(base_path, parse_dates=['Date'])

base_path = os.path.join('Datengrundlage', 'CHFUSD=X.csv')
chf_usd = pd.read_csv(base_path, parse_dates=['Date'])

base_path = os.path.join('Datengrundlage', 'BTC-USD.csv')
btc_usd = pd.read_csv(base_path, parse_dates=['Date'])

gbp_usd.set_index('Date', inplace=True)
eur_usd.set_index('Date', inplace=True)
chf_usd.set_index('Date', inplace=True)
btc_usd.set_index('Date', inplace=True)

fiat_df = pd.concat([gbp_usd['Close'].rename('GBP/USD'),
                     eur_usd['Close'].rename('EUR/USD'),
                     chf_usd['Close'].rename('CHF/USD')], axis=1)
btc_df = btc_usd[['Close']].rename(columns={'Close': 'BTC/USD'})

fiat_log_returns = np.log(fiat_df / fiat_df.shift(1))
btc_log_returns = np.log(btc_df / btc_df.shift(1))

fiat_volatility_total = fiat_log_returns.std()
btc_volatility_total = btc_log_returns.std()

trading_days_per_year = 252
fiat_volatility_annual = fiat_volatility_total * np.sqrt(trading_days_per_year)
btc_volatility_annual = btc_volatility_total * np.sqrt(trading_days_per_year)

print("Volatility over the entire period (4 years):")
print(fiat_volatility_total)
print(btc_volatility_total)

print("\nAnnualized volatility (time period = 4 years):")
print(fiat_volatility_annual)
print(btc_volatility_annual)

plt.figure(figsize=(14, 8))
for column in fiat_df.columns:
    plt.plot(fiat_df.index, np.log(fiat_df[column]), label=column)

plt.title('Logarithmischer Kursverlauf von GBP/USD, EUR/USD und CHF/USD')
plt.xlabel('Datum')
plt.ylabel('Logarithmischer Schlusskurs')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 8))
plt.plot(btc_df.index, np.log(btc_df['BTC/USD']), label='BTC/USD')

plt.title('Logarithmischer Kursverlauf von BTC/USD')
plt.xlabel('Datum')
plt.ylabel('Logarithmischer Schlusskurs')
plt.legend()
plt.grid(True)
plt.show()

fig, ax1 = plt.subplots(figsize=(14, 8))

for column in fiat_df.columns:
    ax1.plot(fiat_df.index, np.log(fiat_df[column]), label=column)

ax1.set_xlabel('Date')
ax1.set_ylabel('Logarithmic closing price Fiat Currencies', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(btc_df.index, np.log(btc_df['BTC/USD']), color='red', label='BTC/USD')

ax2.set_ylabel('Logarithmic closing price BTC/USD')
ax2.legend(loc='upper right')

plt.title('Logarithmic closing price of GBP/USD, EUR/USD, CHF/USD and BTC/USD')
fig.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:20:42 2024

@author: lukas.knoll
"""

import numpy as np
import pandas as pd
import os

#Defining working directory (@Prof. Manshauen & Herr Dietrich Needs to be changed accordingly)
os.chdir(r'C:\Users\Lukas\OneDrive - Provadis Hochschule\Semester 4\Masterthesis\Hauptforschungen\Research_BTC_SA\BTC_SentimentAnalysis')

csv_path = os.path.join('Datengrundlage', 'BTC_SentimentAnalysis_Masters.csv')
btc_data = pd.read_csv(csv_path)

sentiment_scores = btc_data["Vader_Comp"]

def bootstrap_ci(data, num_samples=100000, alpha=0.05):
    boot_samples = []
    for _ in range(num_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        boot_samples.append(np.mean(bootstrap_sample))
    ci_lower = np.percentile(boot_samples, alpha/2 * 100)
    ci_upper = np.percentile(boot_samples, (1-alpha/2) * 100)
    return ci_lower, ci_upper

ci_lower, ci_upper = bootstrap_ci(sentiment_scores)
print("Bootstrap-Konfidenzintervall:", ci_lower, "-", ci_upper)

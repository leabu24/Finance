# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:22:54 2022

@author: leabu
"""

import pandas as pd
import numpy as np

# Definition of variance
def variance(X):
    m = np.mean(X)
    print(m)
    som = 0
    for i in range(len(X)):
        som += (X[i]-m)**2
    return som/len(X)

# Definition of covariance
def covariance(X,Y):
    mx = np.mean(X)
    my = np.mean(Y)
    som = 0
    for i in range(len(X)):
        som += (X[i]-mx)*(Y[i]-my)
    return som/len(X)

# Variation des actions sur 1 an
ADOBE = pd.read_csv("ADBE.csv")
adobe = ADOBE['Adj Close']
var_adobe = [(1-adobe[i+1]/adobe[i]) for i in range(len(adobe)-1)]

META = pd.read_csv("META.csv")
meta = META['Adj Close']
var_meta = [(1-meta[i+1]/meta[i]) for i in range(len(meta)-1)]

MCPA= pd.read_csv("MC.PA.csv")
mcpa = MCPA['Adj Close']
var_mcpa = [(1-mcpa[i+1]/mcpa[i]) for i in range(len(mcpa)-1)]

MSFT = pd.read_csv("MSFT.csv")
msft = MSFT['Adj Close']
var_msft = [(1-msft[i+1]/msft[i]) for i in range(len(msft)-1)]

LLY = pd.read_csv("LLY.csv")
lly = LLY['Adj Close']
var_lly = [(1-lly[i+1]/lly[i]) for i in range(len(lly)-1)]

GOOG = pd.read_csv("GOOG.csv")
goog = GOOG['Adj Close']
var_goog = [(1-goog[i+1]/goog[i]) for i in range(len(goog)-1)]

PYPL = pd.read_csv("PYPL.csv")
pypl = PYPL['Adj Close']
var_pypl = [(1-pypl[i+1]/pypl[i]) for i in range(len(pypl)-1)]

LEAD = pd.read_csv("LEAD.csv")
lead = LEAD['Adj Close']
var_lead = [(1-lead[i+1]/lead[i]) for i in range(len(lead)-1)]

BTC = pd.read_csv("BTC-USD.csv")
btc = BTC['Adj Close']
var_btc = [(1-btc[i+1]/btc[i]) for i in range(len(btc)-1)]

# List of our asset
lst_data = [var_adobe, var_meta, var_mcpa, var_msft, var_lly, var_goog, var_pypl, var_lead, var_btc]
# Weight of each asset in the portfolio
coeff = [2/11,1/11,1/11,1/11,1/11,1/11,1/11,2/11,1/11]

# Number of asset
n = len(lst_data)
# Covariance matrix (init)
M = np.zeros((n,n))

# Covariane matrix 
for i in range(n):
    for j in range(n):
        if i == j: 
            M[i,j] = coeff[i]**2*variance(lst_data[i]) # variance 
        else:
            M[i,j] = coeff[i]*coeff[j]*covariance(lst_data[i],lst_data[j]) # covariance with other asset

# Print covariance matrix
print(M)
# Sum of each line
Sous_tot = np.sum(M, axis=1)
# Sum to get the variance of the portfolio
Tot = np.sum(Sous_tot)

# Standard deviation 
ecart_type = np.sqrt(Tot)
# Print in %
print(ecart_type*100)

# Annual risk
risque_annuel = ecart_type*np.sqrt(12)*100
print("Annual risk =", risque_annuel)
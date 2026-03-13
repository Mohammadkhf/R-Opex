# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 23:51:56 2025

@author: Mohamad
"""

import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

print("===================================")
print("1. BASIC SUMMARY STATISTICS")
print("===================================\n")
print(df.describe(include="all"))  # numeric + categorical summary

print("\n===================================")
print("2. DATA TYPES")
print("===================================\n")
print(df.dtypes)

print("\n===================================")
print("3. MISSING VALUES")
print("===================================\n")
print(df.isna().sum())

# -------------------------------------------------------------
# Identify categorical and numerical columns
# -------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("\n===================================")
print("4. VALUE COUNTS FOR CATEGORICAL FEATURES")
print("===================================\n")
for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col])

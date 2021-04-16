# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Import the training dataset

path = "~/Desktop/Master of Information Management/Master's Thesis/GiveMeSomeCredit/cs-training.csv"
df_training = pd.read_csv(path)

# Exploratory data analysis

df_training.drop(['Unnamed: 0'], axis=1, inplace=True)
df_training.shape
df_training.head()
df_training.info()
df_training.describe()
df_training.columns

## Distribution of data

### class imbalance
df_training['SeriousDlqin2yrs'].mean()
df_training['SeriousDlqin2yrs'].hist(bins=50)
plt.title('SeriousDlqin2yrs')
plt.show()

### there are 1634 people with no income
df_training['MonthlyIncome'].describe()
df_training['MonthlyIncome'].value_counts()
sns.scatterplot(x=np.log(df_training["MonthlyIncome"]), y=np.log(df_training["DebtRatio"]))

### there is one person with age 0
df_training.age.min()
df_training.age.value_counts()

## Outlier detection

### there is a large variety of range in values, some very big
sns.boxplot(df_training['DebtRatio'])
sns.boxplot(df_training['RevolvingUtilizationOfUnsecuredLines'])
sns.boxplot(df_training['NumberOfTime60-89DaysPastDueNotWorse'])
sns.boxplot(df_training['MonthlyIncome'])

## Missing values

df_training.isna().sum()
df_training.isna().sum()/df_training.shape[0]
df_training.age.min()
df_training[df_training['age'] == 0]

## Correlations
corr_matrix = df_training.corr()
corr_matrix
corr_matrix["NumberOfTime30-59DaysPastDueNotWorse"].sort_values(ascending=False)
### High correlation between the PastDue/DaysLate variables (try to solve with VIF<=10)
attributes = ["NumberOfTimes90DaysLate", "NumberOfTime60-89DaysPastDueNotWorse", 
              "NumberOfTime30-59DaysPastDueNotWorse", "SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines",
              "DebtRatio", "NumberOfDependents", "MonthlyIncome", "NumberRealEstateLoansOrLines", "age",
              "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate"]
scatter_matrix(df_training)




















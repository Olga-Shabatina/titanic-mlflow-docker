import pandas as pd
from sklearn.datasets import fetch_openml

titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame

print(df.head())

# preprocess dataset
df = df[['pclass', 'sex', 'age', 'fare', 'survived']].dropna()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# features matrix
X = df[['pclass', 'sex', 'age', 'fare']]
# target vector
y = df['survivev'].astype(int)
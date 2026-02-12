import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame

print(df.head())

# preprocess dataset
df = df[['pclass', 'sex', 'age', 'fare', 'survived']].dropna()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# features matrix
X = df[['pclass', 'sex', 'age', 'fare']]
# target vector
y = df['survived'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# train model
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)
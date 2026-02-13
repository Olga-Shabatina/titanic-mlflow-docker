import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import skops.io as skops
import mlflow
import mlflow.sklearn

RANDOM_FOREST_N_ESTIMATORS = 100

MODEL_PATH = os.getenv("MODEL_PATH", "app/models/model.skops")
FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "false").lower() == "true"

if os.path.exists(MODEL_PATH) and not FORCE_RETRAIN:
    print("Model already exists, skipping training")
    exit(0)
else: os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# load data
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

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Titanic_Classification")

# train model
with mlflow.start_run():
    print(f"Training model: RandomForest with {RANDOM_FOREST_N_ESTIMATORS} trees")
    model = RandomForestClassifier(n_estimators = RANDOM_FOREST_N_ESTIMATORS, random_state = 42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(sk_model=model, name="model")

    skops.dump(model, MODEL_PATH)

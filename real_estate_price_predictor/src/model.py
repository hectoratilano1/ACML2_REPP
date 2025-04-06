
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_model(model, path="model.joblib"):
    joblib.dump(model, path)

def load_model(path="model.joblib"):
    return joblib.load(path)

def predict(model, input_data: pd.DataFrame):
    return model.predict(input_data)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import logging

# Safe fallback logging config in case this file is run directly
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def split_data(X, y):
    logging.info("Splitting data into training and test sets...")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    logging.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model

def save_model(model, path="model.joblib"):
    try:
        joblib.dump(model, path)
        logging.info(f"Model saved to '{path}'")
    except Exception as e:
        logging.error(f"Failed to save model: {e}", exc_info=True)

def load_model(path="model.joblib"):
    try:
        logging.info(f"Loading model from '{path}'...")
        return joblib.load(path)
    except FileNotFoundError:
        logging.error(f"Model file '{path}' not found.")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        raise

def predict(model, input_data: pd.DataFrame):
    logging.info("Making prediction(s)...")
    return model.predict(input_data)

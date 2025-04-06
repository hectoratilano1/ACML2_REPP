
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def prepare_features(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y

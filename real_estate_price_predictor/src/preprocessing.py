import pandas as pd
import streamlit as st

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"❌ File not found: {filepath}")
        st.stop()

def prepare_features(df: pd.DataFrame) -> tuple:
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y

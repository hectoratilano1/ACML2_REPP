
from src import preprocessing, model, evaluate

def run_pipeline():
    # Load and preprocess
    df = preprocessing.load_data("data/final.csv")
    X, y = preprocessing.prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = model.split_data(X, y)

    # Train
    reg_model = model.train_model(X_train, y_train)

    # Save
    model.save_model(reg_model)

    # Evaluate
    mae, rmse, r2 = evaluate.evaluate_model(reg_model, X_test, y_test)
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

if __name__ == "__main__":
    run_pipeline()

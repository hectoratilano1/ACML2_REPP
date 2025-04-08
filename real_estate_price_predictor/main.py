import logging
from src import preprocessing, model, evaluate

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="pipeline.log",  # Logs to a file. Remove to log to terminal instead.
    filemode="w"
)

def run_pipeline():
    try:
        logging.info("Starting real estate price prediction pipeline...")

        # Load and preprocess data
        logging.info("Loading and preprocessing data from final.csv...")
        df = preprocessing.load_data("data/final.csv")
        X, y = preprocessing.prepare_features(df)

        # Split data
        logging.info("Splitting dataset into training and test sets...")
        X_train, X_test, y_train, y_test = model.split_data(X, y)

        # Train model
        logging.info("Training regression model...")
        reg_model = model.train_model(X_train, y_train)

        # Save model
        logging.info("Saving model to model.joblib...")
        model.save_model(reg_model)

        # Evaluate model
        logging.info("Evaluating model performance...")
        mae, rmse, r2 = evaluate.evaluate_model(reg_model, X_test, y_test)

        print(f"✅ MAE: {mae:.2f}")
        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ R²: {r2:.2f}")

        logging.info(f"Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
        print("❌ Something went wrong. Check pipeline.log for details.")

if __name__ == "__main__":
    run_pipeline()

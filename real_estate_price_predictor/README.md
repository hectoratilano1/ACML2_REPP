# Real Estate Price Predictor

This machine learning project predicts real estate prices using Linear Regression based on property features like area, number of bedrooms, and property type.

## Project Structure

```
real_estate_price_predictor/
├── app.py
├── main.py
├── notebooks/
│   └── Real_Estate_Solution.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   └── utils.py
├── data/
│   └── final.csv            # <- Place this file here
├── requirements.txt
└── README.md
```

## How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Place `final.csv` inside the `data/` folder.

3. Run the training pipeline:
   ```bash
   python main.py
   ```

4. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Output

- Model performance: MAE, RMSE, R²
- Web app to estimate property prices interactively
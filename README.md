# Crude Oil Price Prediction Project

This project implements machine learning models to predict crude oil price movements using LSTM (Long Short-Term Memory) neural networks and LightGBM gradient boosting models.

## Project Overview

The project analyzes crude oil price data at 1-minute intervals and builds predictive models to classify whether the price will go up or down in the future. It includes:

- **Data Processing**: Data cleaning and feature engineering from historical crude oil prices
- **LSTM Model**: Deep learning approach using recurrent neural networks
- **LightGBM Model**: Gradient boosting approach for price prediction
- **Backtesting**: Evaluation of model performance on test data
- **Model Comparison**: Side-by-side comparison of both approaches

## Project Structure

```
crude_oil1/
├── backtesting/
│   ├── backtesting_lightgbm.py    # LightGBM model evaluation
│   └── backtesting_lstm.py        # LSTM model evaluation
├── data/
│   ├── crudeoil_1min_last300days.csv  # Historical price data
│   ├── get_data.py                # Data fetching script
│   └── MCX.json                   # Configuration file
├── models/
│   ├── lstm_model.h5              # Trained LSTM model
│   └── lightgbm_model.pkl         # Trained LightGBM model
├── output/
│   ├── lstm.png                   # LSTM results visualization
│   ├── lightgbm.png               # LightGBM results visualization
│   └── compare.png                # Model comparison chart
├── src/
│   ├── lstm_model.py              # LSTM model implementation
│   ├── lightgbm_model.py          # LightGBM model implementation
│   └── compare_models.py          # Comparison script
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dheerendrasolanki03/Project-VCreatek-
cd crude_oil1
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train LSTM Model
```bash
python src/lstm_model.py
```

### Train LightGBM Model
```bash
python src/lightgbm_model.py
```

### Run Backtesting
```bash
python backtesting/backtesting_lstm.py
python backtesting/backtesting_lightgbm.py
```

### Compare Models
```bash
python src/compare_models.py
```

## Features

- **Price Data**: 1-minute interval crude oil prices for the last 300 days
- **Feature Engineering**: 
  - Moving averages (5-day, 10-day)
  - Volatility calculations
  - Price returns
  - Volume analysis

- **Models**:
  - LSTM: 2-3 stacked layers with dropout for regularization
  - LightGBM: Gradient boosting with optimized hyperparameters

## Results

Both models are evaluated on:
- Classification accuracy
- ROC-AUC score
- Confusion matrix
- Detailed classification reports

Comparison visualizations are saved in the `output/` directory.

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework for LSTM
- **LightGBM**: Gradient boosting library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning utilities
- **Matplotlib**: Data visualization

## Author

Dheeendra Solanki

## License

This project is provided as-is for educational purposes.

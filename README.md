# Research Forecasting

This folder contains time-series forecasting experiments (SARIMA, RNN, LSTM, GRU) and notebooks for analysis.

## Setup

### Python environment
Create and activate a virtual environment:

Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install pandas numpy matplotlib scikit-learn statsmodels torch requests
```

### Kaggle token
The scripts that download datasets require a Kaggle API token provided via an environment variable:

Windows (PowerShell):
```
$env:KAGGLE_API_TOKEN="your-token"
```

macOS/Linux:
```
export KAGGLE_API_TOKEN="your-token"
```

## Run

### Main script
```
python main.py
```

### Notebooks
Open and run:
- research_analysis.ipynb
- walmart_sales_forecasting.ipynb

## Notes
- Datasets and generated visualizations are excluded by .gitignore.
- If download fails, verify the Kaggle API token and dataset access.

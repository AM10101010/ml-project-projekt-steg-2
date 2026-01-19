# Store Sales Forecasting with Time Series ML


## 📊 Project Overview

This project demonstrates end-to-end time-series forecasting with a focus on:
- **Data Quality**: Comprehensive validation checks for completeness and consistency
- **Feature Engineering**: Temporal, lag, and calendar-based features with proper handling of data leakage
- **Performance**: Vectorized operations handle 913,000 records in seconds
- **Best Practices**: Modular code structure, temporal train/val split, baseline models

### Dataset
- **Size**: 913,000 records (2013-01-01 to 2017-12-31)
- **Stores**: 10 unique stores
- **Items**: 50 unique items per store
- **Target**: Daily sales count (integer)

## 🎯 Results

### Baseline Model Performance (Validation Set)

| Model                  | MAE   | RMSE  | Description |
|------------------------|-------|-------|-------------|
| Global Mean            | 18.45 | 23.12 | Training set average |
| Lag-1 (yesterday)      | 4.12  | 5.89  | Previous day's sales |
| Rolling Mean 7-day     | 4.08  | 5.76  | 7-day rolling average |

**Validation Period**: October 3, 2017 - December 31, 2017 (90 days, 45,000 samples)

The Rolling Mean 7-day baseline achieves **MAE of 4.08**, establishing a strong benchmark for more complex models.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml-project-projekt-steg-2

# Create virtual environment
python3 -m venv sklearn-env
source sklearn-env/bin/activate  # On Windows: sklearn-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/project_ml_part_2.ipynb
```

Or use the modular Python API:

```python
from src.data import load_sales_data, run_min_checks, temporal_train_val_split
from src.features import create_temporal_features, create_lag_features, create_calendar_features
from src.models import evaluate_baselines

# Load and validate data
df = load_sales_data("Dataset/train.csv")
run_min_checks(df)

# Create features
df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
df = create_temporal_features(df)
df = create_lag_features(df)
df = create_calendar_features(df)

# Train/val split and evaluation
train_df, val_df = temporal_train_val_split(df, val_days=90)
results = evaluate_baselines(train_df, val_df)
```

## 📁 Project Structure

```
.
├── Dataset/
│   └── train.csv                    # Raw sales data (gitignored)
├── src/
│   ├── data/
│   │   ├── loader.py                # Data loading utilities
│   │   ├── validators.py            # Data quality checks
│   │   └── splitters.py             # Train/val splitting
│   ├── features/
│   │   ├── temporal.py              # Time-based features
│   │   ├── lag_features.py          # Lag and rolling features
│   │   └── calendar.py              # Holiday features
│   ├── models/
│   │   └── baselines.py             # Baseline models
│   └── utils/
├── notebooks/
│   └── project_ml_part_2.ipynb      # Main analysis notebook
├── tests/                           # Unit tests (to be added)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── WARP.md                          # Development guide

```

## 🔧 Feature Engineering

### Temporal Features
- **Linear**: Day of week, month, day, quarter, week of year
- **Binary**: Weekend flag, month start/end indicators
- **Cyclical**: Sin/cos transformations for periodic patterns

### Lag Features (per store-item)
- **sales_lag_1**: Previous day's sales
- **sales_lag_7**: Sales from 7 days ago
- **sales_lag_365**: Sales from 365 days ago (seasonal)
- **roll_mean_7**: 7-day rolling average (leakage-free)
- **wow_change**: Week-over-week momentum (%)
- **store_daily_avg_lag1**: Store-level average from previous day

### Calendar Features
- **is_holiday**: Swedish national holidays (via workalendar)

**Critical Note**: All lag features use `.shift()` to prevent data leakage. Features only contain information available at prediction time.

## ⚡ Performance Optimizations

### Vectorized Store-Level Features
The original notebook used `.apply()` with row-wise lambda for `store_daily_avg_lag1`, which was extremely slow on large datasets. The refactored version uses vectorized operations:

**Before** (minutes on 913k rows):
```python
df["store_daily_avg_lag1"] = df.apply(
    lambda r: store_daily.get((r["store"], r["date"] - pd.Timedelta(days=1)), np.nan),
    axis=1  # ⚠️ Slow!
)
```

**After** (seconds on 913k rows):
```python
# Vectorized merge operation - 100x faster
store_daily["date"] = store_daily["date"] + pd.Timedelta(days=1)
df = df.merge(store_daily, on=["store", "date"], how="left")
```

## 🔍 Data Quality Checks

The `run_min_checks()` function validates:
- ✅ Required columns present
- ✅ Type conversions successful
- ✅ No null values or duplicates
- ✅ Cardinality matches expectations (10 stores × 50 items)
- ✅ Per-day completeness (500 records/day)
- ✅ Sales values are non-negative and within bounds

**Example output**:
```
=== Minimal Data Quality Checks ===
Missing columns: []
Type conversion nulls: {'date': 0, 'store': 0, 'item': 0, 'sales': 0}
Key duplicates: 0
Cardinality: {'stores': 10, 'items': 50, 'dates': 1826}
Total rows: 913000 | Total expected: 913000 | Match: True
```

## 📈 Next Steps

1. **ML Models**: Implement Random Forest, XGBoost, LightGBM
2. **Feature Importance**: Analyze which features drive predictions
3. **Per-Store Models**: Train separate models for high-variability items
4. **Hyperparameter Tuning**: Use time-series cross-validation
5. **Model Persistence**: Save trained models with joblib
6. **Error Analysis**: Identify worst-performing store-item combinations

## 🧪 Testing

```bash
# Run unit tests (when implemented)
pytest tests/
```

## 📝 Development Notes

- **Python Version**: 3.14.2
- **Commit Style**: Swedish conventional commits (feat:, fix:, chore:)
- **Date Format**: ISO 8601 (YYYY-MM-DD)
- **Notebook Outputs**: Keep outputs for reproducibility

## 📚 References

- [Workalendar Documentation](https://pypi.org/project/workalendar/)
- [scikit-learn Time Series](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)

## 👤 Author

Data Engineering Portfolio Project - IT Högskolan

## 📄 License

Educational project for portfolio purposes.

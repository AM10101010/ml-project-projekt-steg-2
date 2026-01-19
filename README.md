# Store Sales Forecasting with Time Series ML

A machine learning project for forecasting daily sales across 10 stores and 50 items using time-series features.

## 📊 Project Overview

This project demonstrates end-to-end time-series forecasting with focus on:
- **Data Quality**: Comprehensive validation checks for completeness and consistency
- **Feature Engineering**: Temporal, lag, and calendar-based features with proper data leakage prevention
- **Performance**: Vectorized operations handle 913,000 records in seconds (100x faster than naive approach)
- **Best Practices**: Temporal train/val split, baseline models, documented code

### Dataset
- **Size**: 913,000 records (2013-01-01 to 2017-12-31)
- **Stores**: 10 unique stores
- **Items**: 50 unique items per store
- **Target**: Daily sales count (integer)

## 🎯 Results

### Baseline Model Performance (Validation Set)

| Model                  | MAE   | RMSE  | Description |
|------------------------|-------|-------|-------------|
| Global Mean            | 22.97 | 28.56 | Training set average |
| Lag-1 (yesterday)      | 10.65 | 14.47 | Previous day's sales |
| Rolling Mean 7-day     | 8.64  | 11.37 | 7-day rolling average |

**Validation Period**: October 3, 2017 - December 31, 2017 (90 days, 45,000 samples)

The Rolling Mean 7-day baseline achieves **MAE of 8.64**, establishing a strong benchmark for ML models.

## 🚀 Quick Start

### Local Jupyter

```bash
# Clone the repository
git clone <repository-url>
cd ml-project-projekt-steg-2

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook project_ml_optimized.ipynb
```

### Google Colab

```python
# Clone repository
!git clone https://github.com/YOUR_USERNAME/ml-project-projekt-steg-2.git
%cd ml-project-projekt-steg-2

# Install dependencies
!pip install -r requirements.txt

# Upload dataset
from google.colab import files
import os
os.makedirs('Dataset', exist_ok=True)
uploaded = files.upload()
!mv train.csv Dataset/

# Run the notebook!
```

Or click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/ml-project-projekt-steg-2/blob/main/project_ml_optimized.ipynb)

## 📁 Project Structure

```
.
├── Dataset/
│   └── train.csv              # Sales data (gitignored)
├── project_ml_optimized.ipynb # Main notebook with all code
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Feature Engineering

### Temporal Features
- **Linear**: Day of week, month, day, quarter, week of year
- **Binary**: Weekend flag, month start/end indicators
- **Cyclical**: Sin/cos transformations for periodic patterns (dow, month)

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

## ⚡ Performance Optimization

The notebook includes a **vectorized approach** for store-level features that is **100x faster** than the naive row-wise method:

**Naive approach (slow):**
```python
# Takes 5-10 minutes on 913k rows ❌
df["store_daily_avg_lag1"] = df.apply(
    lambda r: store_daily.get((r["store"], r["date"] - pd.Timedelta(days=1)), np.nan),
    axis=1
)
```

**Vectorized approach (fast):**
```python
# Takes ~1 second ✅
store_daily = df.groupby(['store', 'date'])['sales'].mean().reset_index()
store_daily['date'] = store_daily['date'] + pd.Timedelta(days=1)
df = df.merge(store_daily, on=['store', 'date'], how='left')
```

**Result**: Feature engineering completes in ~1 second instead of 5-10 minutes.

## 🔍 Data Quality Checks

The notebook includes comprehensive validation:
- ✅ Required columns present
- ✅ No null values or duplicates
- ✅ Cardinality matches expectations (10 stores × 50 items)
- ✅ Per-day completeness (500 records/day)
- ✅ Sales values are non-negative and within bounds

## 📈 Next Steps

1. **ML Models**: Implement Random Forest, XGBoost, LightGBM
2. **Feature Importance**: Analyze which features drive predictions
3. **Per-Store Models**: Train separate models for high-variability items
4. **Hyperparameter Tuning**: Use time-series cross-validation
5. **Model Persistence**: Save trained models with joblib

## 📚 Dependencies

- pandas==2.3.3
- numpy==2.4.0
- scikit-learn==1.8.0
- workalendar==17.0.0

See `requirements.txt` for full list.

## 📝 Notes

- **Python Version**: 3.14.2
- **Commit Style**: Conventional commits (feat:, fix:, chore:)
- **Data Format**: ISO 8601 dates (YYYY-MM-DD)

## 👤 Author

Data Engineering Portfolio Project - IT Högskolan

## 📄 License

Educational project for portfolio purposes.

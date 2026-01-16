# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview
This is a machine learning project focused on time-series sales forecasting. The dataset contains daily sales data for 10 stores and 50 items over ~5 years (913,000 records from 2013-2017).

**Dataset structure:**
- `date`: Daily timestamps (YYYY-MM-DD format)
- `store`: Store ID (1-10)
- `item`: Item ID (1-50)
- `sales`: Daily sales count (target variable)

## Environment Setup

### Virtual Environment
The project uses a Python virtual environment in `sklearn-env/`.

**Activate the environment:**
```bash
source sklearn-env/bin/activate
```

**Install packages (if needed):**
```bash
sklearn-env/bin/pip install scikit-learn numpy pandas jupyter ipykernel
```

### Jupyter Notebook
The main work is done in `project_ml_part_2.ipynb`.

**Launch Jupyter:**
```bash
jupyter notebook project_ml_part_2.ipynb
```

Or use your IDE's notebook interface (VSCode, PyCharm, etc.)

## Dataset Access
The training data is located at `Dataset/train.csv` and is gitignored. Features are extracted directly from this file in the notebook.

**Load dataset:**
```python
import pandas as pd
df = pd.read_csv('Dataset/train.csv')
```

## Key ML Architecture

### Feature Engineering Pattern
The notebook implements a systematic feature engineering pipeline for time-series forecasting:

**Time-based features:**
- Day of week, month, day, quarter, week of year
- Weekend indicators, month start/end flags
- Cyclical encodings (sin/cos transformations for day of week and month)

**Historical features (per store-item combination):**
- Lag features: `sales_lag_1` (previous day's sales)
- Rolling aggregates: `roll_mean_7` (7-day rolling average)

**Critical implementation detail:**
All historical features use `.shift(1)` to prevent data leakage. Features are computed per `(store, item)` group after sorting by date.

```python
# Correct pattern to avoid leakage
df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
df['sales_lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
```

### Data Quality Checks
The notebook includes a `run_min_checks()` function that validates:
- Column presence and type conversion
- Null values and duplicates
- Store/item cardinality (should be 10 stores × 50 items)
- Per-day completeness (expected 500 records per day)
- Sales value sanity (non-negative, outlier detection)

**Use this function before training models:**
```python
checks = run_min_checks(df, date_format="%Y-%m-%d")
```

## Development Workflow

### When adding new features:
1. Always sort by `['store', 'item', 'date']` before computing
2. Use `.groupby(['store', 'item'])` for per-timeseries operations
3. Apply `.shift(1)` or appropriate lag to avoid future information
4. Handle NaN values from lag operations (first few rows per group)

### When modifying the notebook:
1. Clear outputs before committing (if needed)
2. Test feature creation on a small subset first
3. Validate no data leakage with the quality checks
4. Document any new features with markdown cells

## Git Configuration
The repository uses Swedish commit messages and follows conventional commit style (feat:, fix:, docs:, etc.).

## Python Environment Notes
- Python 3.14.2 is used
- Core dependencies: scikit-learn 1.8.0, numpy 2.4.0
- No pandas installation detected in venv, but may be available system-wide
- IPython and ipykernel are configured for notebook support

## Important Constraints
- Dataset files (`.csv`, `.pkl`, `.xlsx`) are gitignored
- Model artifacts (`.h5`, `.pkl`, `.joblib`, `.model`) are gitignored
- Keep the notebook self-contained - all data processing should be reproducible from `Dataset/train.csv`

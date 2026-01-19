"""Calendar-based features (holidays, special dates)"""

import pandas as pd
from workalendar.europe import Sweden


def create_calendar_features(df: pd.DataFrame, country: str = "sweden") -> pd.DataFrame:
    """
    Create calendar-based features (holidays, special dates).

    Args:
        df: DataFrame with 'date' column (must be datetime type)
        country: Country for holiday calendar (default: "sweden")

    Returns:
        DataFrame with added features:
        - is_holiday: Binary flag for national holidays

    Note:
        Currently supports Swedish holidays via workalendar.
        Can be extended to support other countries or custom calendars.

    Example:
        >>> df['date'] = pd.to_datetime(df['date'])
        >>> df = create_calendar_features(df)
        >>> print(f"Holidays in dataset: {df['is_holiday'].sum()}")
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' column must be datetime type")

    if country.lower() == "sweden":
        cal = Sweden()
        df["is_holiday"] = df["date"].apply(lambda d: cal.is_holiday(d)).astype(int)
    else:
        raise NotImplementedError(f"Country '{country}' not yet supported")

    return df

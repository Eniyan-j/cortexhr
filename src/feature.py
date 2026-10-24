import pandas as pd


def add_attrition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for the attrition dataset."""
    df = df.copy()

    # Mark negative tenure as missing
    df.loc[df["tenure_months"] < 0, "tenure_months"] = float("nan")
    df["tenure_years"] = df["tenure_months"] / 12

    # Flag raises overdue by more than 12 months
    df["raise_overdue"] = (
        df["months_since_last_raise"] > 12
    ).astype("int8")

    # Mark invalid attendance rates as missing
    valid_attendance = df["attendance_rate_3m"].between(0, 1)
    df.loc[~valid_attendance, "attendance_rate_3m"] = float("nan")
    df["attendance_gap"] = 1 - df["attendance_rate_3m"]

    # Average unpaid leave per month over six months
    df["unpaid_per_month"] = df["unpaid_leave_days_6m"] / 6

    # Monthly overtime load
    df["overtime_load"] = df["avg_overtime_hours_per_month_3m"]

    return df

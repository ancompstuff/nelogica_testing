"""
zigzag_engine.py
----------------
Core swing calculation logic and sensitivity parameter sweeper.
"""

import pandas as pd


def calculate_zigzag(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Computes completed structural ZigZag swings based on point threshold.
    Ignores the final unconfirmed open swing to eliminate calculation bias.
    """
    highs = df['High'].values
    lows = df['Low'].values
    times = df.index

    swings = []
    trend = 0  # 1 = Up Leg, -1 = Down Leg

    last_high = highs[0]
    last_low = lows[0]
    last_high_idx = times[0]
    last_low_idx = times[0]

    for i in range(1, len(df)):
        curr_high = highs[i]
        curr_low = lows[i]
        curr_time = times[i]

        if trend == 0:
            if curr_high >= last_low + threshold:
                trend = 1
                last_high = curr_high
                last_high_idx = curr_time
            elif curr_low <= last_high - threshold:
                trend = -1
                last_low = curr_low
                last_low_idx = curr_time

        elif trend == 1:
            if curr_high > last_high:
                last_high = curr_high
                last_high_idx = curr_time
            elif curr_low <= last_high - threshold:
                swings.append({
                    'type': 'up',
                    'start_time': last_low_idx,
                    'end_time': last_high_idx,
                    'start_price': last_low,
                    'end_price': last_high,
                    'size': last_high - last_low,
                    'bars': len(df.loc[last_low_idx:last_high_idx])
                })
                trend = -1
                last_low = curr_low
                last_low_idx = curr_time

        elif trend == -1:
            if curr_low < last_low:
                last_low = curr_low
                last_low_idx = curr_time
            elif curr_high >= last_low + threshold:
                swings.append({
                    'type': 'down',
                    'start_time': last_high_idx,
                    'end_time': last_low_idx,
                    'start_price': last_high,
                    'end_price': last_low,
                    'size': last_high - last_low,
                    'bars': len(df.loc[last_high_idx:last_low_idx])
                })
                trend = 1
                last_high = curr_high
                last_high_idx = curr_time

    return pd.DataFrame(swings)


def run_automatic_threshold_sweep(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, float]:
    """
    Computes 5-day Average Daily Range (ADR_5d) robustly against NaNs.
    Sweeps thresholds from 2.5% to 25% of ADR_5d and filters out noise.
    """
    daily_high = df.groupby(df.index.date)['High'].max()
    daily_low = df.groupby(df.index.date)['Low'].min()
    daily_ranges = (daily_high - daily_low).dropna()
    daily_ranges = daily_ranges[daily_ranges > 0]

    if len(daily_ranges) == 0:
        raise ValueError("Unable to compute daily ranges: Dataset contains no valid daily OHLC bars.")

    adr_5d = daily_ranges.tail(5).mean()
    total_days = max(len(daily_ranges), 1)

    multipliers = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
    sweep_results = []
    valid_swings_dict = {}

    for mult in multipliers:
        thresh_val = mult * adr_5d
        swings = calculate_zigzag(df, threshold=thresh_val)

        if len(swings) == 0:
            continue

        avg_swings_per_day = len(swings) / total_days
        avg_bars_per_swing = swings['bars'].mean()

        is_valid = (avg_bars_per_swing >= 3.0) and (avg_swings_per_day >= 2.0)

        summary_row = {
            'Pct_ADR': f"{mult * 100:.1f}%",
            'Threshold_Pts': round(thresh_val, 2),
            'Total_Swings': len(swings),
            'Swings_Per_Day': round(avg_swings_per_day, 1),
            'Avg_Bars_Per_Swing': round(avg_bars_per_swing, 1),
            'Mean_Size': round(swings['size'].mean(), 2),
            'Median_Size': round(swings['size'].median(), 2),
            'P75_Size': round(swings['size'].quantile(0.75), 2),
            'P90_Size': round(swings['size'].quantile(0.90), 2),
            'P95_Size': round(swings['size'].quantile(0.95), 2),
            'Status': 'VALID' if is_valid else 'DISCARDED'
        }
        sweep_results.append(summary_row)

        if is_valid:
            valid_swings_dict[f"{mult * 100:.1f}%"] = swings

    sweep_df = pd.DataFrame(sweep_results)
    return sweep_df, valid_swings_dict, adr_5d
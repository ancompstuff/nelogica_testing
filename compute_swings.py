import pandas as pd
import numpy as np


def calculate_zigzag(df: pd.DataFrame, threshold: float = 150.0, mode: str = 'points'):
    """
    Computes completed ZigZag swings for intraday futures data (WINFUT / WDOFUT).

    Parameters:
    - df: DataFrame with ['High', 'Low', 'Close'] indexed by datetime.
    - threshold: Minimum move to register a reversal (e.g., 150 points for WIN, 5 points for WDO).
    - mode: 'points' for absolute point threshold, or 'atr' for multiplier-based threshold.
    """
    if mode == 'atr':
        # Calculate 14-period ATR if dynamic thresholding is desired
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df = df.copy()
        df['threshold'] = atr * threshold
    else:
        df = df.copy()
        df['threshold'] = threshold

    highs = df['High'].values
    lows = df['Low'].values
    times = df.index
    thresholds = df['threshold'].values

    swings = []

    # Initialization
    trend = 0  # 1 for up leg, -1 for down leg
    last_high = highs[0]
    last_low = lows[0]
    last_high_idx = times[0]
    last_low_idx = times[0]

    for i in range(1, len(df)):
        curr_high = highs[i]
        curr_low = lows[i]
        curr_time = times[i]
        curr_thresh = thresholds[i]

        if trend == 0:
            if curr_high >= last_low + curr_thresh:
                trend = 1
                last_high = curr_high
                last_high_idx = curr_time
            elif curr_low <= last_high - curr_thresh:
                trend = -1
                last_low = curr_low
                last_low_idx = curr_time

        elif trend == 1:  # Currently in an Up-swing
            if curr_high > last_high:
                last_high = curr_high
                last_high_idx = curr_time
            elif curr_low <= last_high - curr_thresh:
                # Up-swing closed, save it
                swings.append({
                    'type': 'up',
                    'start_time': last_low_idx,
                    'end_time': last_high_idx,
                    'start_price': last_low,
                    'end_price': last_high,
                    'size': last_high - last_low
                })
                # Reversal to Down-swing
                trend = -1
                last_low = curr_low
                last_low_idx = curr_time

        elif trend == -1:  # Currently in a Down-swing
            if curr_low < last_low:
                last_low = curr_low
                last_low_idx = curr_time
            elif curr_high >= last_low + curr_thresh:
                # Down-swing closed, save it
                swings.append({
                    'type': 'down',
                    'start_time': last_high_idx,
                    'end_time': last_low_idx,
                    'start_price': last_high,
                    'end_price': last_low,
                    'size': last_high - last_low
                })
                # Reversal to Up-swing
                trend = 1
                last_high = curr_high
                last_high_idx = curr_time

    # Converts array of completed swings into DataFrame (ignoring active unclosed swing)
    return pd.DataFrame(swings)


def analyze_swings_and_range(df: pd.DataFrame, swings_df: pd.DataFrame):
    """
    Computes percentile-heavy distribution statistics for Swing Sizes and Daily Ranges.
    """
    # 1. Swing Statistics
    sizes = swings_df['size']

    stats = {
        'Count': len(sizes),
        'Min': sizes.min(),
        'P25': sizes.quantile(0.25),
        'Median (P50)': sizes.median(),
        'Mean': sizes.mean(),
        'P75': sizes.quantile(0.75),
        'P90': sizes.quantile(0.90),
        'P95': sizes.quantile(0.95),
        'Max': sizes.max(),
    }

    # 2. Daily Range Statistics (High - Low of each session)
    daily_df = df.groupby(df.index.date).agg({'High': 'max', 'Low': 'min'})
    daily_ranges = daily_df['High'] - daily_df['Low']

    daily_stats = {
        'Count': len(daily_ranges),
        'Min': daily_ranges.min(),
        'P25': daily_ranges.quantile(0.25),
        'Median (P50)': daily_ranges.median(),
        'Mean': daily_ranges.mean(),
        'P75': daily_ranges.quantile(0.75),
        'P90': daily_ranges.quantile(0.90),
        'P95': daily_ranges.quantile(0.95),
        'Max': daily_ranges.max(),
    }

    summary_table = pd.DataFrame([stats, daily_stats], index=['Swing Size', 'Daily Range']).T
    return summary_table

# --- Usage Example ---
# df = pd.read_csv('WINFUT_5m.csv', parse_dates=['Datetime'], index_col='Datetime')

# Recommended base thresholds:
# WINFUT: threshold=150.0 (points)
# WDOFUT: threshold=5.0 (points)

# swings = calculate_zigzag(df, threshold=150.0, mode='points')
# stats_table = analyze_swings_and_range(df, swings)
# print(stats_table)
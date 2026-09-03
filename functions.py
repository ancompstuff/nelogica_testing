import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ReportLab imports for automated PDF compilation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ==============================================================================
# MODULE 1: DATA LOADER & PARSER
# ==============================================================================

import os
import pandas as pd


def read_b3_data(ticker: str, timeframe: str = '5min') -> pd.DataFrame:
    """
    Reads Nelogica B3 historical CSV exports using native Brazilian format parsing.

    Parameters:
    - ticker: 'WINFUT', 'WDOFUT', 'PETR4', 'VALE3', 'ITUB4'
    - timeframe: '1min' or '5min'
    """
    data_directories = {
        '1min': r"F:\Documents\PyCharmProjects\GitHub\TradingData\Data_files\Nelogica_Historical_data\1_min_Data",
        '5min': r"F:\Documents\PyCharmProjects\GitHub\TradingData\Data_files\Nelogica_Historical_data\5_min_Data"
    }

    if timeframe not in data_directories:
        raise ValueError(f"Invalid timeframe '{timeframe}'. Must be '1min' or '5min'.")

    # Determine filename suffix: Futures (_F_0_), Equities (_B_0_)
    suffix = "_F_0_" if "FUT" in ticker.upper() else "_B_0_"
    filename = f"{ticker.upper()}{suffix}{timeframe}.csv"
    full_path = os.path.join(data_directories[timeframe], filename)

    if not os.path.exists(full_path):
        if os.path.exists(filename):
            full_path = filename
        else:
            raise FileNotFoundError(f"Target data file not found: {full_path}")

    # Read directly using native Brazilian locale settings (decimal=',', thousands='.')
    df = pd.read_csv(
        full_path,
        sep=';',
        encoding='latin1',
        decimal=',',
        thousands='.'
    )

    # Clean whitespace from column headers
    df.columns = df.columns.str.strip()

    # Standardize column headers to English schema
    df = df.rename(columns={
        'Abertura': 'Open',
        'Máximo': 'High',
        'Mínimo': 'Low',
        'Fechamento': 'Close',
        'Quantidade': 'Volume_Qty'
    })

    # Clean whitespace from date and time strings and construct Datetime index
    data_str = df['Data'].astype(str).str.strip()
    hora_str = df['Hora'].astype(str).str.strip()

    df['Datetime'] = pd.to_datetime(
        data_str + ' ' + hora_str,
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )

    # Set Datetime index and enforce chronological ascending order
    df = df.dropna(subset=['Datetime']).set_index('Datetime').sort_index()

    # Return pure numeric OHLCV DataFrame
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


# ==============================================================================
# MODULE 2: ZIGZAG & SENSITIVITY SWEEPER
# ==============================================================================

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
                # Up-swing complete: log swing
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
                # Down-swing complete: log swing
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
    Computes 5-day Average Daily Range (ADR_5d) robustly against NaNs or corrupt rows.
    Sweeps thresholds from 2.5% to 25% of ADR_5d and filters out noise.
    """
    # ROBUST ADR CALCULATION: Compute max high and min low per session explicitly dropping NaNs
    daily_high = df.groupby(df.index.date)['High'].max()
    daily_low = df.groupby(df.index.date)['Low'].min()
    daily_ranges = (daily_high - daily_low).dropna()

    # Exclude invalid zero or negative ranges
    daily_ranges = daily_ranges[daily_ranges > 0]

    if len(daily_ranges) == 0:
        raise ValueError("Unable to compute daily ranges: Dataset contains no valid daily OHLC bars.")

    # Calculate 5-day ADR
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

        # Noise filter rule: minimum 3 bars per swing & 2 swings per day
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


# ==============================================================================
# MODULE 3: REPORTLAB PDF PRESENTATION GENERATOR
# ==============================================================================

def generate_pdf_report(symbol: str, timeframe: str, sweep_df: pd.DataFrame, valid_swings: dict, adr_5d: float,
                        output_filename: str):
    """
    Generates a PDF analysis document containing visual distribution charts and parameter sweep tables.
    """
    chart_image_path = f"temp_chart_{symbol}_{timeframe}.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"{symbol} ({timeframe}) - Swing Distribution & Sensitivity Analysis", fontsize=14, fontweight='bold')

    valid_mask = sweep_df['Status'] == 'VALID'
    axes[0].bar(sweep_df['Pct_ADR'], sweep_df['Swings_Per_Day'],
                color=np.where(valid_mask, '#1f77b4', '#d62728'), alpha=0.85)
    axes[0].set_title("Swings Per Day (Blue = Valid, Red = Discarded)")
    axes[0].set_xlabel("% of 5-Day ADR")
    axes[0].set_ylabel("Frequency (Swings / Day)")
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    baseline_key = "10.0%" if "10.0%" in valid_swings else (list(valid_swings.keys())[0] if valid_swings else None)

    if baseline_key:
        baseline_swings = valid_swings[baseline_key]['size']
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        values = np.percentile(baseline_swings, percentiles)

        axes[1].plot(percentiles, values, marker='o', color='#2ca02c', linewidth=2)
        axes[1].set_title(f"Swing Size Quantiles ({baseline_key} ADR Baseline)")
        axes[1].set_xlabel("Percentile Distribution")
        axes[1].set_ylabel("Swing Size (Points)")
        axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(chart_image_path, dpi=200)
    plt.close()

    doc = SimpleDocTemplate(output_filename, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36,
                            bottomMargin=36)
    styles = getSampleStyleSheet()

    custom_title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1A2B4C'),
        spaceAfter=10
    )

    story = [
        Paragraph(f"<b>Statistical Swing Analysis Report: {symbol} ({timeframe})</b>", custom_title_style),
        Paragraph(f"<b>5-Day Average Daily Range (ADR_5d):</b> {adr_5d:.2f} pts", styles['Normal']),
        Spacer(1, 15),
        RLImage(chart_image_path, width=540, height=243),
        Spacer(1, 15),
        Paragraph("<b>Parameter Sweep & Noise Filtering Matrix</b>", styles['Heading2']),
        Spacer(1, 5)
    ]

    table_data = [[col.replace('_', ' ') for col in sweep_df.columns]]
    for _, row in sweep_df.iterrows():
        table_data.append([str(val) for val in row.values])

    col_widths = [45, 55, 55, 55, 60, 50, 50, 50, 50, 50, 55]
    table = Table(table_data, colWidths=col_widths)

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A2B4C')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
    ]))

    story.append(table)
    doc.build(story)

    if os.path.exists(chart_image_path):
        os.remove(chart_image_path)


# ==============================================================================
# MODULE 4: AUTOMATED BATCH EXECUTION LOOP
# ==============================================================================

if __name__ == "__main__":
    # List of target assets and timeframes to process
    symbols = ['WINFUT', 'WDOFUT', 'PETR4', 'VALE3', 'ITUB4']
    timeframes = ['5min', '1min']

    print("=== STARTING B3 STATISTICAL SWING COMPUTATION PIPELINE ===")

    for symbol in symbols:
        for tf in timeframes:
            try:
                print(f"\nProcessing: {symbol} | Timeframe: {tf}...")

                # 1. Read CSV data
                df = read_b3_data(ticker=symbol, timeframe=tf)

                # 2. Run sensitivity parameter sweep
                sweep_df, valid_swings, adr_5d = run_automatic_threshold_sweep(df)

                # 3. Print results table
                print(f"5-Day Average Daily Range (ADR_5d): {adr_5d:.2f} pts")
                print(sweep_df.to_string(index=False))

                # 4. Export PDF Report
                pdf_name = f"{symbol}_{tf}_swing_analysis.pdf"
                generate_pdf_report(symbol, tf, sweep_df, valid_swings, adr_5d, pdf_name)
                print(f" PDF Report saved: {pdf_name}")

            except FileNotFoundError as e:
                print(f"  Skipping {symbol} ({tf}): File not found at expected location.")
            except Exception as e:
                print(f"  Error processing {symbol} ({tf}): {str(e)}")

    print("\n=== PIPELINE EXECUTION COMPLETE ===")

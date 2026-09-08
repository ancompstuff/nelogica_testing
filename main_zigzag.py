"""
main_zigzag.py
--------------
Batch execution entry point for ZigZag swing sensitivity analysis.
"""

import os
from data_loader import read_b3_data
from zigzag_engine import run_automatic_threshold_sweep
from zigzag_pdf_generator import generate_zigzag_pdf_report


def run_zigzag_pipeline():
    output_dir = os.path.join("results", "zigzag")

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

                # 4. Export PDF Report to results/zigzag/
                pdf_name = f"{symbol}_{tf}_zigzag_analysis.pdf"
                generate_zigzag_pdf_report(
                    symbol=symbol,
                    timeframe=tf,
                    sweep_df=sweep_df,
                    valid_swings=valid_swings,
                    adr_5d=adr_5d,
                    output_filename=pdf_name,
                    output_dir=output_dir
                )
                print(f" PDF Report saved: {os.path.join(output_dir, pdf_name)}")

            except FileNotFoundError:
                print(f"  Skipping {symbol} ({tf}): File not found at expected location.")
            except Exception as e:
                print(f"  Error processing {symbol} ({tf}): {str(e)}")

    print("\n=== PIPELINE EXECUTION COMPLETE ===")


if __name__ == "__main__":
    run_zigzag_pipeline()
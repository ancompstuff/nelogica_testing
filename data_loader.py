"""
data_loader.py
--------------
Reusable data ingestion module for Nelogica B3 CSV historical exports.
"""

import os
import pandas as pd

# Default directory configurations
DATA_DIRECTORIES = {
    '1min': r"F:\Documents\PyCharmProjects\GitHub\TradingData\Data_files\Nelogica_Historical_data\1_min_Data",
    '5min': r"F:\Documents\PyCharmProjects\GitHub\TradingData\Data_files\Nelogica_Historical_data\5_min_Data"
}


def read_b3_data(ticker: str, timeframe: str = '5min', data_dirs: dict = None) -> pd.DataFrame:
    """
    Reads Nelogica B3 historical CSV exports using native Brazilian format parsing.

    Parameters:
    -----------
    ticker : str
        Asset ticker symbol (e.g., 'WINFUT', 'WDOFUT', 'PETR4', 'VALE3', 'ITUB4')
    timeframe : str
        Target timeframe ('1min' or '5min')
    data_dirs : dict, optional
        Custom mapping for timeframe paths. Defaults to DATA_DIRECTORIES.

    Returns:
    --------
    pd.DataFrame
        Cleaned OHLCV DataFrame indexed by Datetime.
    """
    dirs = data_dirs or DATA_DIRECTORIES

    if timeframe not in dirs:
        raise ValueError(f"Invalid timeframe '{timeframe}'. Must be one of {list(dirs.keys())}.")

    # Determine filename suffix: Futures (_F_0_), Equities (_B_0_)
    suffix = "_F_0_" if "FUT" in ticker.upper() else "_B_0_"
    filename = f"{ticker.upper()}{suffix}{timeframe}.csv"
    full_path = os.path.join(dirs[timeframe], filename)

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
        'Quantidade': 'Volume'
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
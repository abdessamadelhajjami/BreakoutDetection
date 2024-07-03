import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import requests
from io import BytesIO
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import requests
import tempfile
import gzip
import shutil
import subprocess
import warnings




# Telegram bot configuration
TELEGRAM_API_URL = "https://api.telegram.org/bot7010066680:AAHJxpChwtfiK0PBhJFAGCgn6sd4HVOVARI/sendMessage"
TELEGRAM_CHAT_ID = "-1002197712630"

def send_telegram_message(message):
    response = requests.post(
        TELEGRAM_API_URL,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message}
    )
    print(f"Telegram response: {response.json()}")

def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return df["Symbol"].tolist()

def download_sp500_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

def table_exists(conn, schema, table_name):
    query = f"""
    SELECT COUNT(*)
    FROM information_schema.tables 
    WHERE table_schema = '{schema}'
    AND table_name = '{table_name.upper()}';
    """
    result = conn.cursor().execute(query).fetchone()[0]
    return result > 0

def get_last_date(conn, schema, table_name):
    if not table_exists(conn, schema, table_name):
        return '2010-01-01'
    
    query = f'SELECT MAX("Date") FROM "{schema}"."{table_name}"'
    cursor = conn.cursor()
    cursor.execute(query)
    last_date = cursor.fetchone()[0]
    cursor.close()
    
    if last_date is None:
        return '2010-01-01'
    else:
        return (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

def create_table_if_not_exists(conn, schema, table_name):
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS "{schema}"."{table_name.upper()}" (
            "Date" DATE, 
            "Open" FLOAT, 
            "High" FLOAT, 
            "Low" FLOAT, 
            "Close" FLOAT, 
            "Adj_Close" FLOAT, 
            "Volume" FLOAT
        )
    """)
    cursor.close()

def load_data_to_snowflake(conn, df, schema, table_name):
    create_table_if_not_exists(conn, schema, table_name)
    df.reset_index(inplace=True)
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df['Date'] = df['Date'].astype(str)
    success, nchunks, nrows, _ = write_pandas(conn, df, table_name.upper())
    return success, nchunks, nrows

def calculate_pivot_reversals(df, window=3):
    pivot_series = pd.Series([0]*len(df), index=df.index)
    for candle in range(window, len(df) - window):
        pivotHigh, pivotLow = True, True
        current_high, current_low = df.iloc[candle]['High'], df.iloc[candle]['Low']
        for i in range(candle-window, candle+window+1):
            if df.iloc[i]['Low'] < current_low:
                pivotLow = False
            if df.iloc[i]['High'] > current_high:
                pivotHigh = False
        if pivotHigh and pivotLow:
            pivot_series[candle] = 3  
        elif pivotHigh:
            pivot_series[candle] = 2
        elif pivotLow:
            pivot_series[candle] = 1
    return pivot_series

def collect_channel(df, candle, backcandles, window=1):
    localdf = df[candle-backcandles-window:candle-window]
    highs, idxhighs = localdf[localdf['SAR Reversals'] == 1].High.values, localdf[localdf['SAR Reversals'] == 1].High.index
    lows, idxlows = localdf[localdf['SAR Reversals'] == 2].Low.values, localdf[localdf['SAR Reversals'] == 2].Low.index
    if len(lows) >= 2 and len(highs) >= 2:
        sl_lows, interc_lows, sl_highs, interc_highs, _, _ = stats.linregress(idxhighs, highs)
        sl_highs, interc_highs, _, _, _ = stats.linregress(idxlows, lows)
        return sl_lows, interc_lows, sl_highs, interc_highs, 0, 0
    else:
        return 0, 0, 0, 0, 0, 0

def line_crosses_candles(data, slope, intercept, start_index, end_index):
    body_crosses = 0
    for i in range(start_index, end_index + 1):
        candle = data.iloc[i]
        predicted_price = intercept + slope * (i - start_index)
        open_price, close_price = candle['Open'], candle['Close']
        body_high, body_low = max(open_price, close_price), min(open_price, close_price)
        if body_low <= predicted_price <= body_high:
            body_crosses += 1
    return body_crosses > 1

def isBreakOut(df, candle, window=1):
    for backcandles in [14, 20, 40, 60]:  
        if (candle - backcandles - window) < 0:
            continue
        try:
            sl_lows, interc_lows, sl_highs, interc_highs, _, _ = collect_channel(df, candle, backcandles, window)
            if sl_lows == 0 and sl_highs == 0:
                continue
        except:
            continue
        prev_idx, curr_idx = candle - 1, candle
        prev_high, prev_low, prev_close = df.iloc[prev_idx]['High'], df.iloc[prev_idx]['Low'], df.iloc[prev_idx]['Close']
        curr_high, curr_low, curr_close, curr_open = df.iloc[candle]['High'], df.iloc[candle]['Low'], df.iloc[candle]['Close'], df.iloc[candle]['Open']
        if (not line_crosses_candles(df, sl_highs, interc_highs, candle-backcandles, candle-1) and 
            prev_low < (sl_highs * prev_idx + interc_highs) and
            prev_close > (sl_highs * prev_idx + interc_highs) and
            curr_open > (sl_highs * curr_idx + interc_highs) and
            curr_close > (sl_highs * curr_idx + interc_highs)):
            return 2, sl_highs, interc_highs
        if (not line_crosses_candles(df, sl_lows, interc_lows, candle-backcandles, candle-1) and
            prev_high > (sl_lows * prev_idx + interc_lows) and
            prev_close < (sl_lows * prev_idx + interc_lows) and
            curr_open < (sl_lows * curr_idx + interc_lows) and
            curr_close < (sl_lows * curr_idx + interc_lows)):
            return 1, sl_lows, interc_lows
    return 0, None, None

def calculate_all_indicators(df):
    df['Pivot_Reversals'] = calculate_pivot_reversals(df)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    mid_band = df['Close'].rolling(window=20).mean()
    sd = df['Close'].rolling(window=20).std()
    df['Bollinger_High'] = mid_band + (2 * sd)
    df['Bollinger_Low'] = mid_band - (2 * sd)
    df['Bollinger_Mid'] = mid_band
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Keltner_Mid'] = df['Close'].ewm(span=20, adjust=False).mean()
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(window=20).mean()
    df['Keltner_High'] = df['Keltner_Mid'] + 2 * df['ATR']
    df['Keltner_Low'] = df['Keltner_Mid'] - 2 * df['ATR']
    
    # Add Slope and Intercept calculation for all rows
    slopes = []
    intercepts = []
    breakout_types = []
    for i in range(len(df)):
        breakout_type, slope, intercept = isBreakOut(df, i)
        slopes.append(slope)
        intercepts.append(intercept)
        breakout_types.append(breakout_type)
    df['Slope'] = slopes
    df['Intercept'] = intercepts
    df['Breakout_Type'] = breakout_types
    
    return df

def extract_and_flatten_features(df, candle):
    if candle < 14:
        return np.array([])
    data_window = df.iloc[candle-14:candle]
    normalized_data = pd.DataFrame()
    for period in [7, 20, 50, 200]:
        sma_key = f'SMA_{period}'
        normalized_data[f'Norm_{sma_key}'] = data_window[sma_key] / data_window['Close']
    normalized_data['Norm_MACD'] = (data_window['MACD'] - data_window['MACD'].mean()) / data_window['MACD'].std()
    normalized_data['Norm_RSI'] = (data_window['RSI'] - data_window['RSI'].mean()) / data_window['RSI'].std()
    normalized_data['Norm_Bollinger_Width'] = (data_window['Bollinger_High'] - data_window['Bollinger_Low']) / data_window['Bollinger_Mid']
    normalized_data['Norm_Volume'] = data_window['Volume'] / data_window['Volume_MA']
    normalized_data['Norm_Keltner_High'] = (data_window['Keltner_High'] - data_window['Keltner_Mid']) / data_window['Keltner_Mid']
    normalized_data['Norm_Keltner_Low'] = (data_window['Keltner_Low'] - data_window['Keltner_Mid']) / data_window['Keltner_Mid']
    normalized_data['Slope'] = df['Slope'].iloc[candle]
    normalized_data['Intercept'] = df['Intercept'].iloc[candle]
    normalized_data['Breakout_Type'] = df['Breakout_Type'].iloc[candle]
    flattened_features = normalized_data.values.flatten().tolist()
    flattened_features.extend([normalized_data['Slope'].iloc[-1], normalized_data['Intercept'].iloc[-1], normalized_data['Breakout_Type'].iloc[-1]])
    return np.array(flattened_features)


import subprocess

def download_model_from_snowflake(account, user, stage, model_file, local_dir):
    # Construire la commande SnowSQL GET
    get_command = f'snowsql -a {account} -u {user} -q "GET @{stage}/{model_file} file://{local_dir}"'

    # Exécuter la commande GET
    subprocess.run(get_command, shell=True, check=True)

def load_model():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model = joblib.load('OHLCV_DATA_TTWO_model.pkl')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None



def main():
    SP500_CONN = {
        'account': 'MOODBPJ-ATOS_AWS_EU_WEST_1',
        'user': 'AELHAJJAMI',
        'password': 'Abdou3012',
        'warehouse': 'COMPUTE_WH',
        'database': 'BREAKOUDETECTIONDB',
        'schema': 'SP500',
    }

    print('[MAIN] : Connecting to Snowflake for SP500 data...')
    conn = snowflake.connector.connect(
        user=SP500_CONN['user'],
        password=SP500_CONN['password'],
        account=SP500_CONN['account'],
        warehouse=SP500_CONN['warehouse'],
        database=SP500_CONN['database'],
        schema=SP500_CONN['schema']
    )
    print('[MAIN] : Connected to Snowflake for SP500 data.')

    symbol = 'TTWO'
    table_name = f'ohlcv_data_{symbol}'.upper()
    query = f'SELECT * FROM {SP500_CONN["schema"]}.{table_name}'
    
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error reading data from Snowflake: {e}")
        return

    df = calculate_all_indicators(df)
    today_idx = df.index[-1]
    breakout_type, slope, intercept = isBreakOut(df, today_idx)

    df.loc[today_idx, 'Slope'] = slope
    df.loc[today_idx, 'Intercept'] = intercept
    df.loc[today_idx, 'Breakout_Type'] = breakout_type
    
    print(f"Breakout type today for {symbol} is: {breakout_type}")
    breakout_type = 1  # Pour test
    slope = -0.2  # Pour test
    intercept = -0.11  # Pour test
    
    if breakout_type > 0:
        print("YEPP1")
        features = extract_and_flatten_features(df, today_idx)
        if features.size == 0:
            print("Features extraction failed.")
            return

        model_filename = "OHLCV_DATA_TTWO_model.pkl"
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                model = joblib.load(model_filename)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        print("YEEP2")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(1, -1))
        prediction = model.predict(features_scaled)
        prediction = ['VB']  # Pour le test, on force la prédiction à 'VB'
        
        if prediction[0] in ['VH', 'VB']:
            print("YEEP3")
            message = f"A True Bullish/Bearish breakout detected today for {symbol}: {prediction[0]}"
            send_telegram_message(message)
    print("finish")
    conn.close()

if __name__ == "__main__":
    main()


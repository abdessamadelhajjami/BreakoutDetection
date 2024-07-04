import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import snowflake.connector
from snowflake.snowpark import Session
from snowflake.connector.pandas_tools import write_pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import os


# Telegram bot configuration
TELEGRAM_API_URL = "https://api.telegram.org/bot7010066680:AAHJxpChwtfiK0PBhJFAGCgn6sd4HVOVARI/sendMessage"
TELEGRAM_CHAT_ID = "https://t.me/Breakout_Channel" 

# Snowflake connection configuration
SNOWFLAKE_CONN = {
    'account': 'MOODBPJ-ATOS_AWS_EU_WEST_1',
        'user': 'AELHAJJAMI',
        'password': 'Abdou3012',
        'warehouse': 'CRYPTO_WH',
        'database': 'BREAKOUDETECTIONDB',
        'schema': 'SP500',
}


# Function to get SP500 components
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return df["Symbol"].tolist()

# Download SP500 data
def download_sp500_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

# Load data into Snowflake
def load_data_to_snowflake(data, table_name):
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_CONN['user'],
        password=SNOWFLAKE_CONN['password'],
        account=SNOWFLAKE_CONN['account'],
        warehouse=SNOWFLAKE_CONN['warehouse'],
        database=SNOWFLAKE_CONN['database'],
        schema=SNOWFLAKE_CONN['schema']
    )
    df = data.reset_index()
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df['Date'] = df['Date'].astype(str)
    create_table_if_not_exists(conn, table_name)
    write_pandas(conn, df, table_name)
    conn.close()

def create_table_if_not_exists(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS "{SNOWFLAKE_CONN['schema']}"."{table_name}" (
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

# Check if table exists
def table_exists(conn, table_name):
    query = f"""
    SELECT COUNT(*)
    FROM information_schema.tables 
    WHERE table_schema = '{SNOWFLAKE_CONN['schema']}'
    AND table_name = '{table_name.upper()}';
    """
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()[0]
    cursor.close()
    return result > 0

# Get the last date from the table
def get_last_date(conn, table_name):
    if not table_exists(conn, table_name):
        return '2010-01-01'
    
    query = f'SELECT MAX("Date") FROM "{SNOWFLAKE_CONN["schema"]}"."{table_name}"'
    cursor = conn.cursor()
    cursor.execute(query)
    last_date = cursor.fetchone()[0]
    cursor.close()
    
    if last_date is None:
        return '2010-01-01'
    else:
        return (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

# Calculate pivot reversals
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

# Collect channel information
def collect_channel(df, candle, backcandles, window=1):
    localdf = df[candle-backcandles-window:candle-window]
    highs, idxhighs = localdf[localdf['SAR_Reversals'] == 1].High.values, localdf[localdf['SAR_Reversals'] == 1].High.index
    lows, idxlows = localdf[localdf['SAR_Reversals'] == 2].Low.values, localdf[localdf['SAR_Reversals'] == 2].Low.index
    if len(lows) >= 2 and len(highs) >= 2:
        sl_lows, interc_lows, _, _, _ = stats.linregress(idxlows, lows)
        sl_highs, interc_highs, _, _, _ = stats.linregress(idxhighs, highs)
        return sl_lows, interc_lows, sl_highs, interc_highs, 0, 0
    else:
        return 0, 0, 0, 0, 0, 0

# Check if the line crosses candles
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

# Detect breakouts
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

# Confirm breakout
def confirm_breakout(df, breakout_index, confirmation_candles=5, threshold_percentage=2):
    if breakout_index + confirmation_candles >= len(df):
        return None, None
    breakout_type = df.loc[breakout_index, 'Breakout_Type']
    breakout_price = df.loc[breakout_index, 'Intercept']
    last_confirmed_price = df.iloc[breakout_index + confirmation_candles]['Close']
    price_variation_percentage = ((last_confirmed_price - breakout_price) / breakout_price) * 100
    if breakout_type == 1:
        if price_variation_percentage <= -threshold_percentage:
            return 'VB', price_variation_percentage
        else:
            return 'FB', price_variation_percentage
    elif breakout_type == 2:
        if price_variation_percentage >= threshold_percentage:
            return 'VH', price_variation_percentage
        else:
            return 'FH', price_variation_percentage

def calculate_sma(df, periods):
    for period in periods:
        sma_key = f'SMA_{period}'
        df[sma_key] = df['Close'].rolling(window=period).mean()
    return df

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    exp1 = df['Close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_signal'] = signal
    return df

def calculate_rsi(df, period=14):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_bbands(df, period=20, std_dev=2):
    mid_band = df['Close'].rolling(window=period).mean()
    sd = df['Close'].rolling(window=period).std()
    df['Bollinger_High'] = mid_band + (std_dev * sd)
    df['Bollinger_Low'] = mid_band - (std_dev * sd)
    df['Bollinger_Mid'] = mid_band
    return df

def calculate_volume_ma(df, period=20):
    df['Volume_MA'] = df['Volume'].rolling(window=period).mean()
    return df

def calculate_keltner_channel(df, ema_period=20, atr_period=20, multiplier=2):
    df['Keltner_Mid'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(window=atr_period).mean()
    df['Keltner_High'] = df['Keltner_Mid'] + multiplier * df['ATR']
    df['Keltner_Low'] = df['Keltner_Mid'] - multiplier * df['ATR']
    return df

# Calculate indicators
def calculate_all_indicators(df):
    df = calculate_sma(df, [7, 20, 50, 200])
    df = calculate_macd(df)
    df = calculate_rsi(df)
    df = calculate_bbands(df)
    df = calculate_volume_ma(df)
    df = calculate_keltner_channel(df)
    return df

# Extract and flatten features
def extract_and_flatten_features(candle, df):
    if candle < 14:
        return None
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

# Detect and label breakouts
def detect_and_label_breakouts(df):
    Breakout_indices = []
    Breakout_confirmed = []
    Breakout_percentage = []

    for index in df.index:
        if df.loc[index, 'Breakout_Type'] in [1, 2]:
            result = confirm_breakout(df, index)
            if result:
                confirmation_label, variation = result
                df.at[index, 'Breakout_Confirmed'] = confirmation_label
                df.at[index, 'Price_Variation_Percentage'] = variation
                Breakout_indices.append(index)
                Breakout_confirmed.append(confirmation_label)
                Breakout_percentage.append(variation)
    return df

# Train and save the model
def train_and_save_model(session, table_name):
    df = session.table(table_name).to_pandas()
    df = calculate_all_indicators(df)
    df['SAR_Reversals'] = calculate_pivot_reversals(df)
    results = [isBreakOut(df, i) for i in range(len(df))]
    df['Breakout_Type'] = [r[0] for r in results]
    df['Slope'] = [r[1] for r in results]
    df['Intercept'] = [r[2] for r in results]
    df = detect_and_label_breakouts(df)
    Breakout_indices = df[df['Breakout_Confirmed'].notna()].index
    features = []
    labels = []
    for index in Breakout_indices:
        flat_features = extract_and_flatten_features(index, df)
        if flat_features is not None:
            features.append(flat_features)
            labels.append(df.loc[index, 'Breakout_Confirmed'])
    if not features:
        print(f"No valid data to train for {table_name}")
        return
    X, y = np.array(features), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=800, max_depth=10, random_state=42, n_jobs=1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"Accuracy on test data for {table_name}: {accuracy}")
    print(f"Classification Report for {table_name}:\n{report}")
    model_filename = f"{table_name}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")


# Send Telegram notification
def send_telegram_message(message):
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    response = requests.post(TELEGRAM_API_URL, data=payload)
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")

# Main function
def main():
    connection_parameters = {
        'account': SNOWFLAKE_CONN['account'],
        'user': SNOWFLAKE_CONN['user'],
        'password': SNOWFLAKE_CONN['password'],
        'warehouse': SNOWFLAKE_CONN['warehouse'],
        'database': SNOWFLAKE_CONN['database'],
        'schema': SNOWFLAKE_CONN['schema']
    }
    session = Session.builder.configs(connection_parameters).create()
    # symbols = get_sp500_components()
    # end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # for symbol in symbols:
    #     table_name = f'ohlcv_data_{symbol}'.upper()
    #     conn = snowflake.connector.connect(
    #         user=SNOWFLAKE_CONN['user'],
    #         password=SNOWFLAKE_CONN['password'],
    #         account=SNOWFLAKE_CONN['account'],
    #         warehouse=SNOWFLAKE_CONN['warehouse'],
    #         database=SNOWFLAKE_CONN['database'],
    #         schema=SNOWFLAKE_CONN['schema']
    #     )
    #     last_date = get_last_date(conn, table_name)
    #     conn.close()
    #     data = download_sp500_data(symbol, last_date, end_date)
    #     if not data.empty:
    #         load_data_to_snowflake(data, table_name)

    symbol = 'AAPL'
    table_name = f'ohlcv_data_{symbol}'.upper()
    tables = session.sql(f"SELECT DISTINCT table_name FROM information_schema.tables WHERE table_schema = '{SNOWFLAKE_CONN['schema']}'").collect()

    for table in tables:
        train_and_save_model(session, table['TABLE_NAME'])
        # Check for VH or VB
        df = session.table(table['TABLE_NAME']).to_pandas()
        vh_vb = df[(df['Breakout_Confirmed'] == 'VH') | (df['Breakout_Confirmed'] == 'VB')]
        for _, row in vh_vb.iterrows():
            message = f"A True Bullish/Bearish breakout detected today for the action {table['TABLE_NAME']}: {row['Breakout_Confirmed']} on {row['Date']}"
            send_telegram_message(message)
    session.close()

if __name__ == "__main__":
    main()

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

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
    data.reset_index(inplace=True)
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
            "Volume" FLOAT,
            "SAR_Reversals" FLOAT,
            "Breakout_Type" FLOAT,
            "Slope" FLOAT,
            "Intercept" FLOAT,
            "Breakout_Confirmed" STRING,
            "Norm_SMA_7" FLOAT,
            "Norm_SMA_20" FLOAT,
            "Norm_SMA_50" FLOAT,
            "Norm_SMA_200" FLOAT,
            "Norm_MACD" FLOAT,
            "Norm_RSI" FLOAT,
            "Norm_Bollinger_Width" FLOAT,
            "Norm_Volume" FLOAT,
            "Norm_Keltner_High" FLOAT,
            "Norm_Keltner_Low" FLOAT
        )
    """)
    cursor.close()

def load_data_to_snowflake(conn, df, schema, table_name):
    create_table_if_not_exists(conn, schema, table_name)
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df['Date'] = df['Date'].astype(str)
    success, nchunks, nrows, _ = write_pandas(conn, df, table_name.upper())
    return success, nchunks, nrows

# 2. Calculer les points pivots et les ajouter à la base de données.
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
    df['SAR_Reversals'] = pivot_series
    return df

# 3. Détecter les breakouts et ajouter les colonnes slope, intercept, et breakout_type à la base de données.
def collect_channel(df, candle, backcandles, window=1):
    localdf = df[candle-backcandles-window:candle-window]
    highs, idxhighs = localdf[localdf['SAR_Reversals'] == 1].High.values, localdf[localdf['SAR_Reversals'] == 1].High.index
    lows, idxlows = localdf[localdf['SAR_Reversals'] == 2].Low.values, localdf[localdf['SAR_Reversals'] == 2].Low.index
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

def detect_breakouts(df):
    results = [isBreakOut(df, i) for i in range(len(df))]
    df['Breakout_Type'] = [r[0] for r in results]
    df['Slope'] = [r[1] for r in results]
    df['Intercept'] = [r[2] for r in results]
    return df

# 4. Calculer les 13 features et les ajouter à la base de données.
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
def extract_and_flatten_features(df, candle):
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
    breakout_indices = []
    breakout_confirmed = []
    breakout_percentage = []

    for index in df.index:
        if df.loc[index, 'Breakout_Type'] in [1, 2]:
            result = confirm_breakout(df, index)
            if result:
                confirmation_label, variation = result
                df.at[index, 'Breakout_Confirmed'] = confirmation_label
                df.at[index, 'Price_Variation_Percentage'] = variation
                breakout_indices.append(index)
                breakout_confirmed.append(confirmation_label)
                breakout_percentage.append(variation)
    return df

# Fonction d'entraînement et d'enregistrement du modèle sur la VM
def train_and_save_model(df, table_name):
    df = calculate_all_indicators(df)
    df['SAR_Reversals'] = calculate_pivot_reversals(df)
    df = detect_breakouts(df)
    df = detect_and_label_breakouts(df)
    breakout_indices = df[df['Breakout_Confirmed'].notna()].index
    features = []
    labels = []
    for index in breakout_indices:
        flat_features = extract_and_flatten_features(df, index)
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

def calculate_and_save_features(conn, schema, table_name):
    query = f'SELECT * FROM {schema}.{table_name}'
    df = pd.read_sql(query, conn)
    df = calculate_all_indicators(df)
    df['SAR Reversals'] = calculate_pivot_reversals(df)
    results = [isBreakOut(df, i) for i in range(len(df))]
    df['Breakout_Type'] = [r[0] for r in results]
    df['Slope'] = [r[1] for r in results]
    df['Intercept'] = [r[2] for r in results]
    df = detect_and_label_breakouts(df)

    # Enregistrer les données avec les features calculées dans Snowflake
    create_table_if_not_exists(conn, schema, table_name + '_FEATURES')
    df.reset_index(inplace=True)
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df['Date'] = df['Date'].astype(str)
    success, nchunks, nrows, _ = write_pandas(conn, df, table_name + '_FEATURES')
    return success, nchunks, nrows

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

    symbol = 'AAPL'
    table_name = f'ohlcv_data_{symbol}'.upper()
    
    start_date = get_last_date(conn, SP500_CONN['schema'], table_name)
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    data = download_sp500_data(symbol, start_date, end_date)

    if not data.empty:
        success, nchunks, nrows = load_data_to_snowflake(conn, data, SP500_CONN['schema'], table_name)
        print(f"Data loaded to Snowflake: {success}, {nchunks}, {nrows} rows")

    query = f'SELECT * FROM {SP500_CONN["schema"]}.{table_name}'
    df = pd.read_sql(query, conn)

    calculate_and_save_features(conn, SP500_CONN['schema'], table_name)
    train_and_save_model(df, table_name)

if __name__ == "__main__":
    main()

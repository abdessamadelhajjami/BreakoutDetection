import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import joblib
from sklearn.preprocessing import StandardScaler
import requests
import os 
import tempfile


# Telegram bot configuration
TELEGRAM_API_URL = "https://api.telegram.org/bot7010066680:AAHJxpChwtfiK0PBhJFAGCgn6sd4HVOVARI/sendMessage"
TELEGRAM_CHAT_ID = "-1002197712630"

# Snowflake connection configuration
SNOWFLAKE_CONN = {
    'account': 'MOODBPJ-ATOS_AWS_EU_WEST_1',
    'user': 'AELHAJJAMI',
    'password': 'Abdou3012',
    'warehouse': 'COMPUTE_WH',
    'database': 'BREAKOUDETECTIONDB',
    'schema': 'SP500',
}

# Updated function to get SP500 components excluding 'BRK.B'
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    symbols = df["Symbol"].tolist()
    if "BRK.B" in symbols:
        symbols.remove("BRK.B") 
    if "BF.B" in symbols:
        symbols.remove("BF.B")
    return symbols

def download_sp500_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

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
        return (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

def create_table_if_not_exists(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS "{SNOWFLAKE_CONN['schema']}"."{table_name.upper()}" (
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

def load_data_to_snowflake(conn, df, table_name):
    create_table_if_not_exists(conn, table_name)
    
    df.reset_index(inplace=True)
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df['Date'] = df['Date'].astype(str)
    
    success, nchunks, nrows, _ = write_pandas(conn, df, f"{SNOWFLAKE_CONN['schema']}.{table_name.upper()}")
    return success, nchunks, nrows
    
# Detect breakout using isBreakOut
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


# Collect channel information
def collect_channel(df, candle, backcandles, window=1):
    localdf = df[candle-backcandles-window:candle-window]
    highs, idxhighs = localdf[localdf['SAR Reversals'] == 1].High.values, localdf[localdf['SAR Reversals'] == 1].High.index
    lows, idxlows = localdf[localdf['SAR Reversals'] == 2].Low.values, localdf[localdf['SAR Reversals'] == 2].Low.index
    if len(lows) >= 2 and len(highs) >= 2:
        sl_lows, interc_lows, sl_highs, interc_highs, _, _ = stats.linregress(idxlows, lows)
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
    # Vérifiez que l'index de la bougie est un entier
    if isinstance(candle, pd.Timestamp):
        candle = candle.to_pydatetime().date()
    
    # Assurez-vous que 'candle' est un entier
    if isinstance(candle, pd.DatetimeIndex):
        candle = candle[-1].to_pydatetime().date()

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
    normalized_data['Breakout_Type'] = df['Breakout Type'].iloc[candle]

    flattened_features = normalized_data.values.flatten().tolist()
    return np.array(flattened_features)





# Send Telegram notification
def send_telegram_message(message):
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    response = requests.post(TELEGRAM_API_URL, data=payload)
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")



def read_data_from_snowflake(conn, table_name):
    query = f'SELECT * FROM "{SNOWFLAKE_CONN["schema"]}"."{table_name}"'
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    cursor.close()
    return df

def main():
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_CONN['user'],
        password=SNOWFLAKE_CONN['password'],
        account=SNOWFLAKE_CONN['account'],
        warehouse=SNOWFLAKE_CONN['warehouse'],
        database=SNOWFLAKE_CONN['database'],
        schema=SNOWFLAKE_CONN['schema']
    )
    
    symbols = get_sp500_components()
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    for symbol in symbols:
        print(f"Processing {symbol}")
        table_name = f'ohlcv_data_{symbol}'.upper()
        last_date = get_last_date(conn, table_name)
        # data = download_sp500_data(symbol, last_date, end_date)
        # if not data.empty:
        #     success, nchunks, nrows = load_data_to_snowflake(conn, data, table_name)
        #     print(f"Data loaded: {success}, {nchunks} chunks, {nrows} rows")
        # else:
        #     print(f"No new data for {symbol}")

        # Vérifier les breakouts pour aujourd'hui
        df = read_data_from_snowflake(conn, table_name)
        if df.empty:
            print("base de donnée empy")
            continue

        df = calculate_all_indicators(df)
        today_idx = df.index[-1]
        breakout_type, slope, intercept = isBreakOut(df, today_idx)
        df['Breakout Type'], df['Slope'], df['Intercept'] = zip(*[isBreakOut(df, i) for i in range(len(df))])

        breakout_type = 1
        print("breakout type today is :", breakout_type)
        breakout_type = 1
        slope = 1.23
        intercept= 0.25
        if breakout_type > 0:
            print("YEPP1")
            features = extract_and_flatten_features(df, today_idx)
            if features.size == 0:
                continue

            model_filename = f"{table_name}_model.pkl"

            # Create a temporary directory
            with tempfile.TemporaryDirectory() as tmpdirname:
                local_model_path = os.path.join(tmpdirname, model_filename)
                
                conn.cursor().execute(f"USE DATABASE YAHOOFINANCEDATA")
                conn.cursor().execute(f"GET @STOCK_DATA.INTERNAL_STAGE/{model_filename} file://{local_model_path}")
            
                model = joblib.load(local_model_path)
                print("YEEP2")
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features.reshape(1, -1))
                prediction = model.predict(features_scaled)
                
                if prediction[0] in ['VH', 'VB']:
                    print("YEEP3")
                    message = f"A True Bullish/Bearish breakout detected today for {symbol}: {prediction[0]}"
                    send_telegram_message(message)
        print("finish")

    conn.close()

if __name__ == "__main__":
    main()

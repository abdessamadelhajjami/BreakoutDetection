import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import snowflake.connector
import snowflake.snowpark as snowpark
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import requests
import os

# Configuration du bot Telegram
TELEGRAM_API_URL = "https://api.telegram.org/bot7010066680:AAHJxpChwtfiK0PBhJFAGCgn6sd4HVOVARI/sendMessage"
TELEGRAM_CHAT_ID = "-1002197712630"  # Utilisez l'ID de votre chat

# Fonction pour envoyer un message Telegram
def send_telegram_message(message):
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }
    response = requests.post(TELEGRAM_API_URL, data=payload)
    if response.status_code == 200:
        print("Message envoyé avec succès")
    else:
        print(f"Échec de l'envoi du message: {response.text}")

# Snowflake connection configuration
SNOWFLAKE_CONN = {
    'account': 'MOODBPJ-ATOS_AWS_EU_WEST_1',
    'user': 'AELHAJJAMI',
    'password': 'Abdou3012',
    'warehouse': 'CRYPTO_WH',
    'database': 'BREAKOUDETECTIONDB',
    'schema': 'SP500',
}

# Functions to get SP500 components
def get_sp500_components():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return df["Symbol"].tolist()

# Download SP500 data
def download_sp500_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

# Load data into Snowflake
def load_data_to_snowflake(conn, df, table_name):
    df.reset_index(inplace=True)
    cursor = conn.cursor()

    # Créer la table si elle n'existe pas
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            Date DATE, 
            Open FLOAT, 
            High FLOAT, 
            Low FLOAT, 
            Close FLOAT, 
            Adj_Close FLOAT, 
            Volume FLOAT
        )
    """)

    # Insérer les données dans la table
    success, nchunks, nrows, _ = snowflake.connector.pandas_tools.write_pandas(conn, df, table_name)

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
    highs, idxhighs = localdf[localdf['SAR Reversals'] == 1].High.values, localdf[localdf['SAR Reversals'] == 1].High.index
    lows, idxlows = localdf[localdf['SAR Reversals'] == 2].Low.values, localdf[localdf['SAR Reversals'] == 2].Low.index
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

# Fonction principale
def main():
    connection_parameters = {
        'account': SNOWFLAKE_CONN['account'],
        'user': SNOWFLAKE_CONN['user'],
        'password': SNOWFLAKE_CONN['password'],
        'warehouse': SNOWFLAKE_CONN['warehouse'],
        'database': SNOWFLAKE_CONN['database'],
        'schema': SNOWFLAKE_CONN['schema']
    }
    session = snowpark.Session.builder.configs(connection_parameters).create()
    symbols = get_sp500_components()
    start_date = '2010-01-01'
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    for symbol in symbols:
        print(f"Processing {symbol}")
        table_name = f"ohlcv_data_{symbol}"

        # Check if table exists and get the max date
        result = session.sql(f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = '{SNOWFLAKE_CONN['schema']}' 
              AND table_name = '{table_name.upper()}'
        """).collect()

        if result[0][0] > 0:
            max_date_result = session.sql(f"""
                SELECT MAX(Date) FROM {SNOWFLAKE_CONN['schema']}.{table_name}
            """).collect()
            max_date = max_date_result[0][0]
            start_date = (pd.Timestamp(max_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        # Download data
        data = download_sp500_data(symbol, start_date, end_date)
        data.reset_index(inplace=True)

        if not data.empty:
            load_data_to_snowflake(session, data, table_name)
        else:
            print(f"No new data for {symbol}")

    tables = session.sql(f"SELECT DISTINCT table_name FROM information_schema.tables WHERE table_schema = '{SNOWFLAKE_CONN['schema']}'").collect()
    for table in tables:
        df = session.table(table['TABLE_NAME']).to_pandas()
        df = calculate_all_indicators(df)
        df['SAR Reversals'] = calculate_pivot_reversals(df)
        results = [isBreakOut(df, i) for i in range(len(df))]
        df['Breakout Type'] = [r[0] for r in results]
        df['Slope'] = [r[1] for r in results]
        df['Intercept'] = [r[2] for r in results]

        # Charger le modèle
        model_filename = f"{table['TABLE_NAME']}_model.pkl"
        session.file.get(f"@YAHOOFINANCEDATA.STOCK_DATA.INTERNAL_STAGE/{model_filename}", model_filename)
        model = joblib.load(model_filename)

        # Préparer les données pour la prédiction
        for i in range(14, len(df)):
            if df.loc[i, 'Breakout Type'] in [1, 2]:
                flat_features = extract_and_flatten_features(i, df)
                if flat_features is not None:
                    features = np.array(flat_features).reshape(1, -1)
                    imputer = SimpleImputer(strategy='mean')
                    features = imputer.fit_transform(features)
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    prediction = model.predict(features_scaled)
                    if prediction in ['VH', 'VB']:
                        message = f"A True {'Bullish' if prediction == 'VH' else 'Bearish'} breakout detected for the action {table['TABLE_NAME']} on {df.loc[i, 'Date']}"
                        send_telegram_message(message)
    session.close()

if __name__ == "__main__":
    main()


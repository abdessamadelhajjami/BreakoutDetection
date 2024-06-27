import yfinance as yf
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

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
        return (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

# Create table if it does not exist
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

# Load data into Snowflake using `write_pandas`
def load_data_to_snowflake(conn, df, table_name):
    # Create the table if it doesn't exist
    create_table_if_not_exists(conn, table_name)
    
    # Reset the index
    df.reset_index(inplace=True)
    
    # Rename columns to avoid invalid identifiers
    df.columns = [col.replace(' ', '_') for col in df.columns]
    
    # Insert data using `write_pandas`
    success, nchunks, nrows, _ = write_pandas(conn, df, table_name.upper())
    return success, nchunks, nrows

# Main function
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
        data = download_sp500_data(symbol, last_date, end_date)
        if not data.empty:
            success, nchunks, nrows = load_data_to_snowflake(conn, data, table_name)
            print(f"Data loaded: {success}, {nchunks} chunks, {nrows} rows")
        else:
            print(f"No new data for {symbol}")

    conn.close()

if __name__ == "__main__":
    main()


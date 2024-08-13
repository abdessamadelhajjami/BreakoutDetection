# Breakout Detection System

This project is designed to detect and analyze breakout patterns in the S&P 500 stock data using various technical indicators and machine learning. The system downloads historical stock data, calculates technical indicators, detects breakout patterns, trains a machine learning model to predict future breakouts, and sends notifications via a Telegram bot.

## Features

- Downloads S&P 500 stock data using Yahoo Finance API.
- Calculates technical indicators such as SMA, MACD, RSI, Bollinger Bands, Keltner Channels, and Volume Moving Average.
- Detects pivot reversals and breakout patterns.
- Stores and processes data using Snowflake.
- Trains and saves a Random Forest model to predict breakout confirmations.
- Sends breakout alerts via Telegram.

## Requirements

- Python 3.7 or higher
- Libraries: `yfinance`, `pandas`, `numpy`, `scipy`, `snowflake-connector-python`, `snowflake-snowpark-python`, `scikit-learn`, `joblib`, `requests`

## Setup

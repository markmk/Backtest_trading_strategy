import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import empyrical as ep
import os  # Import os module

# Function to fetch 1-hour data
def fetch_hourly_data(symbol, period='2y'):
    data = yf.download(symbol, period=period, interval='1h')
    data.dropna(inplace=True)
    return data


def split_data(data, split_ratio=0.5):
    split_point = int(len(data) * split_ratio)
    in_sample_data = data.iloc[:split_point]
    out_of_sample_data = data.iloc[split_point:]
    return in_sample_data, out_of_sample_data


# Backtesting
def backtest_strategy(data, drop_threshold=0.1, holding_period=4):
    data = data.copy()
    data['Return'] = data['Close'].pct_change() * 100  # Percentage return
    data['Signal'] = 0

    # Generate signals
    for i in range(1, len(data)):
        # Check for significant drop
        if data['Return'].iloc[i] <= -drop_threshold:
            data['Signal'].iloc[i] = 1  # Buy signal

    # Extract trades
    trades = []
    for i in range(len(data)):
        if data['Signal'].iloc[i] == 1:
            entry_price = data['Close'].iloc[i]
            entry_time = data.index[i]

            # Determine exit
            exit_index = i + holding_period
            if exit_index >= len(data):
                exit_index = len(data) - 1  # Prevent index out of range
            exit_price = data['Close'].iloc[exit_index]
            exit_time = data.index[exit_index]

            # Calculate profit
            profit = exit_price - entry_price
            profit_pct = (profit / entry_price) * 100

            trades.append({
                'Symbol': symbol,
                'Entry_Time': entry_time,
                'Entry_Price': entry_price,
                'Exit_Time': exit_time,
                'Exit_Price': exit_price,
                'Profit': profit,
                'Profit_%': profit_pct
            })

    trades_df = pd.DataFrame(trades)

    # Performance Metrics
    total_trades = len(trades_df)
    profitable_trades = trades_df[trades_df['Profit'] > 0]
    total_profitable_trades = len(profitable_trades)
    percent_profitable = (total_profitable_trades / total_trades) * 100 if total_trades > 0 else 0
    average_profit_per_trade = trades_df['Profit'].mean() if total_trades > 0 else 0
    highest_single_trade_profit = trades_df['Profit'].max() if total_trades > 0 else 0

    performance = {
        'Symbol': symbol,
        'Total Trades': total_trades,
        'Total Profitable Trades': total_profitable_trades,
        'Percent Profitable Trades': percent_profitable,
        'Average Profit Per Trade': average_profit_per_trade,
        'Highest Single Trade Profit': highest_single_trade_profit
    }

    return trades_df, performance

# List of currency pairs
currency_pairs = [
    'EURUSD=X',
    'GBPUSD=X',
    'USDJPY=X',
    'AUDUSD=X',
    'USDCAD=X',
    'USDCHF=X',
    'NZDUSD=X',
    'EURGBP=X',
    'EURJPY=X',
    'GBPJPY=X'
]

# Directory to save data files
data_dir = 'currency_data'
os.makedirs(data_dir, exist_ok=True)

# Run backtests
all_performance = []
all_trades = []

for symbol in currency_pairs:
    print(f"Processing {symbol}...")
    try:
        data = fetch_hourly_data(symbol)
        if data.empty:
            print(f"No data for {symbol}")
            continue

        # Save fetched data to CSV
        data.to_csv(os.path.join(data_dir, f"{symbol}_data.csv"))

        # Split the data into in-sample and out-of-sample
        in_sample_data, out_of_sample_data = split_data(data)

        # Backtest on in-sample data
        trades_in_sample, performance_in_sample = backtest_strategy(in_sample_data)
        performance_in_sample['Symbol'] = symbol
        performance_in_sample['Period'] = 'In-Sample'

        # Backtest on out-of-sample data
        trades_out_sample, performance_out_sample = backtest_strategy(out_of_sample_data)
        performance_out_sample['Symbol'] = symbol
        performance_out_sample['Period'] = 'Out-of-Sample'

        # Append performance metrics
        all_performance.append(performance_in_sample)
        all_performance.append(performance_out_sample)

        # Append trades
        if not trades_in_sample.empty:
            trades_in_sample['Symbol'] = symbol
            trades_in_sample['Period'] = 'In-Sample'
            all_trades.append(trades_in_sample)
        if not trades_out_sample.empty:
            trades_out_sample['Symbol'] = symbol
            trades_out_sample['Period'] = 'Out-of-Sample'
            all_trades.append(trades_out_sample)

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

# Compile performance summary
performance_df = pd.DataFrame(all_performance)
performance_df = performance_df[['Symbol', 'Period', 'Total Trades', 'Total Profitable Trades',
                                 'Percent Profitable Trades', 'Average Profit Per Trade',
                                 'Highest Single Trade Profit']]

print("\nBacktesting Performance Summary:")
print(performance_df)


if all_trades:
    all_trades_df = pd.concat(all_trades, ignore_index=True)
    # Save trades to CSV
    all_trades_df.to_csv("all_trades.csv", index=False)

# Save performance summary to CSV
performance_df.to_csv("performance_summary.csv", index=False)

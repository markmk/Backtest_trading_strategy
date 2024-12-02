import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Define date ranges
in_sample_start = '2010-01-01'
in_sample_end = '2018-12-31'
out_sample_start = '2022-09-01'
out_sample_end = '2023-10-31'

# Define the list of symbols for each strategy
symbols_tam = ['SPY', 'QQQ', 'DIA', 'IWM']  # Indices ETFs for Turnaround Monday Strategy
symbols_nih = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']  # Tech stocks for New Intraday High Strategy

# Initialize lists to store results
list_trades_tam = []
list_trades_nih = []


# Function to fetch data with error handling
def fetch_intraday_data(symbol, start_date, end_date, interval):
    all_data = []
    current_start = datetime.strptime(start_date, '%Y-%m-%d')
    while current_start < datetime.strptime(end_date, '%Y-%m-%d'):
        current_end = min(current_start + timedelta(days=730), datetime.strptime(end_date, '%Y-%m-%d'))
        print(f"Fetching {interval} data for {symbol} from {current_start} to {current_end}...")
        try:
            data = yf.download(
             symbol, 
             start=current_start.strftime('%Y-%m-%d'), 
             end=current_end.strftime('%Y-%m-%d'), 
             interval=interval, 
             progress=False
            )
            if not data.empty:
                all_data.append(data)
            else:
                print(f"No data returned for {symbol} from {current_start} to {current_end}.")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
        current_start = current_end + timedelta(days=1)
    return pd.concat(all_data) if all_data else pd.DataFrame()


# Function to process Turnaround Monday Strategy
def process_turnaround_monday(symbol):
    # Fetch daily data for in-sample period
    data_daily = fetch_intraday_data(symbol, in_sample_start, in_sample_end, '1d')
    if data_daily.empty:
        return pd.DataFrame()  # Return empty DataFrame if data is empty

    data_daily['Return'] = data_daily['Close'].pct_change()
    data_daily['Weekday'] = data_daily.index.weekday  # Monday=0, Friday=4

    # Identify negative Fridays
    data_daily['Friday Negative'] = (data_daily['Return'] < 0) & (data_daily['Weekday'] == 4)
    data_daily['Buy Signal'] = data_daily['Friday Negative'].shift(1) & (data_daily['Weekday'] == 0)

    # Simulate Trades for in-sample
    trades_in_sample = simulate_trades_tam(data_daily, symbol, 'In-Sample')

    # Fetch hourly data for out-of-sample period
    data_hourly = fetch_intraday_data(symbol, out_sample_start, out_sample_end, '1h')
    if data_hourly.empty:
        return trades_in_sample  # Return only in-sample trades if out-of-sample data is empty

    data_hourly = data_hourly.copy()
    data_hourly['Date'] = data_hourly.index.date
    data_hourly['Hour'] = data_hourly.index.hour
    data_hourly['Weekday'] = pd.to_datetime(data_hourly['Date']).weekday # Monday=0, Friday=4
    data_hourly['Return'] = data_hourly['Close'].pct_change()

    # Identify negative last hours on Friday
    friday_last_hour = data_hourly[(data_hourly['Weekday'] == 4) & (data_hourly['Hour'] == 15)].copy()
    friday_last_hour['Friday Negative'] = friday_last_hour['Return'] < 0

    # Ensure the indices align
    friday_last_hour = friday_last_hour.reset_index()
    monday_first_hour = data_hourly[(data_hourly['Weekday'] == 0) & (data_hourly['Hour'] == 9)].copy()
    monday_first_hour = monday_first_hour.reset_index()

    # Merge Friday signals with Monday data
    if len(friday_last_hour) > 0:
        # Align the signals
        friday_negative_values = friday_last_hour['Friday Negative'].values
        monday_first_hour['Buy Signal'] = np.roll(friday_negative_values, shift=1)
        monday_first_hour['Buy Signal'] = monday_first_hour['Buy Signal'].fillna(False)
    else:
        monday_first_hour['Buy Signal'] = False

    # Set index back to datetime for consistency
    monday_first_hour.set_index('Datetime', inplace=True)

    # Simulate Trades for out-of-sample
    trades_out_sample = simulate_trades_tam(monday_first_hour, symbol, 'Out-of-Sample', hourly=True)

    return pd.concat([trades_in_sample, trades_out_sample], ignore_index=True)

# Function to simulate trades for Turnaround Monday Strategy
def simulate_trades_tam(data, symbol, sample_period, hourly=False):
    trades = []
    buy_signals = data[data['Buy Signal']]
    for idx, row in buy_signals.iterrows():
        entry_date = idx
        if hourly:
            exit_date = idx + pd.Timedelta(hours=3)  # Exit after 3 hours
        else:
            exit_date = idx + pd.Timedelta(days=3)  # Exit after 3 days

        # Adjust for market hours
        if exit_date not in data.index:
            # Find the closest available date
            dates = data.index
            if len(dates) == 0:
                continue
            exit_date = min(dates, key=lambda d: abs(d - exit_date))

        entry_price = data.loc[entry_date]['Close']
        exit_price = data.loc[exit_date]['Close']

        return_pct = (exit_price - entry_price) / entry_price * 100
        trades.append({
            'Symbol': symbol,
            'Entry Date': entry_date,
            'Exit Date': exit_date,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Return (%)': return_pct,
            'Sample': sample_period
        })
    return pd.DataFrame(trades)

# Process Turnaround Monday Strategy for all symbols
for symbol in symbols_tam:
    trades = process_turnaround_monday(symbol)
    if not trades.empty:
        list_trades_tam.append(trades)

# Concatenate all trades into a single DataFrame
df_trades_tam = pd.concat(list_trades_tam, ignore_index=True) if list_trades_tam else pd.DataFrame()

# Function to process New Intraday High Strategy
def process_new_intraday_high(symbol):
    # Fetch daily data for in-sample period
    data_daily = fetch_intraday_data(symbol, in_sample_start, in_sample_end, '1d')
    if data_daily.empty:
        return pd.DataFrame()

    data_daily['Previous High'] = data_daily['High'].shift(1)
    data_daily['Buy Signal'] = data_daily['Open'] > data_daily['Previous High']

    # Simulate Trades for in-sample
    trades_in_sample = simulate_trades_nih(data_daily, symbol, 'In-Sample')

    # Fetch hourly data for out-of-sample period
    data_hourly = fetch_intraday_data(symbol, out_sample_start, out_sample_end, '1h')
    if data_hourly.empty:
        return trades_in_sample

    data_hourly = data_hourly.copy()
    data_hourly['Previous High'] = data_hourly['High'].shift(1)
    data_hourly['Buy Signal'] = data_hourly['Open'] > data_hourly['Previous High']

    # Simulate Trades for out-of-sample
    trades_out_sample = simulate_trades_nih(data_hourly, symbol, 'Out-of-Sample', hourly=True)

    return pd.concat([trades_in_sample, trades_out_sample], ignore_index=True)

# Function to simulate trades for New Intraday High Strategy
def simulate_trades_nih(data, symbol, sample_period, hourly=False):
    trades = []
    buy_signals = data[data['Buy Signal']]
    for idx, row in buy_signals.iterrows():
        entry_date = idx
        entry_price = row['Open']
        exit_price = row['Close']  # Exit at the close of the same period
        return_pct = (exit_price - entry_price) / entry_price * 100
        trades.append({
            'Symbol': symbol,
            'Entry Date': entry_date,
            'Exit Date': entry_date,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Return (%)': return_pct,
            'Sample': sample_period
        })
    return pd.DataFrame(trades)

# Process New Intraday High Strategy for all symbols
for symbol in symbols_nih:
    trades = process_new_intraday_high(symbol)
    if not trades.empty:
        list_trades_nih.append(trades)

# Concatenate all trades into a single DataFrame
df_trades_nih = pd.concat(list_trades_nih, ignore_index=True) if list_trades_nih else pd.DataFrame()

# Function to calculate performance metrics
def calculate_performance_metrics(data, initial_portfolio_value):
    if data.empty:
        return {
            "Final Portfolio Value": initial_portfolio_value,
            "Total Trades": 0,
            "Average Return/Trade": 0,
            "Percentage Profitable": 0,
            "Win Rate": 0
        }

    data = data.sort_values('Entry Date').reset_index(drop=True)
    data['Portfolio Value'] = initial_portfolio_value * (1 + data['Return (%)'] / 100).cumprod()
    final_portfolio_value = data['Portfolio Value'].iloc[-1]
    total_trades = len(data)
    average_return = data['Return (%)'].mean()
    profitable_trades = len(data[data['Return (%)'] > 0])
    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

    return {
        "Final Portfolio Value": final_portfolio_value,
        "Total Trades": total_trades,
        "Average Return/Trade": average_return,
        "Percentage Profitable": (profitable_trades / total_trades) * 100,
        "Win Rate": win_rate
    }

# Function to generate performance summary
def generate_performance_summary(df_trades, strategy_name):
    if df_trades.empty:
        print(f"No trades were executed for {strategy_name}.")
        return pd.DataFrame()

    performance_summaries = []

    # Split data by Sample and Symbol
    grouped = df_trades.groupby(['Sample', 'Symbol'])
    for (sample, symbol), group in grouped:
        metrics = calculate_performance_metrics(group, initial_portfolio_value=10000)
        performance_summaries.append({
            'Sample': sample,
            'Symbol': symbol,
            **metrics
        })

    performance_df = pd.DataFrame(performance_summaries)
    print(f"\nPerformance Summary - {strategy_name}:")
    print(performance_df)
    return performance_df

# Generate performance summaries for both strategies
print("========== Turnaround Monday Strategy ==========")
summary_tam = generate_performance_summary(df_trades_tam, "Turnaround Monday Strategy")

print("\n========== New Intraday High Strategy ==========")
summary_nih = generate_performance_summary(df_trades_nih, "New Intraday High Strategy")

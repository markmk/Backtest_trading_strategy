import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch data
ticker = 'SPY'
start_date = '2000-01-01'
end_date = '2023-10-09'
data = yf.download(ticker, start=start_date, end=end_date)

# Prepare data
data.reset_index(inplace=True)
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Prev_Close'] = data['Close'].shift(1)

# Identify Friday's close for Mondays
def get_friday_close(row):
    if row['DayOfWeek'] == 0:
        friday_date = row['Date'] - pd.Timedelta(days=3)
        friday_close = data.loc[data['Date'] == friday_date, 'Close']
        if not friday_close.empty:
            return friday_close.values[0]
    return np.nan

data['Friday_Close'] = data.apply(get_friday_close, axis=1)

# Generate signals
data['Signal'] = np.where(
    (data['DayOfWeek'] == 0) & (data['Close'] < data['Friday_Close']), 1, 0)

# Extract trades
entries = data[data['Signal'] == 1][['Date', 'Close']]
entries.rename(columns={'Date': 'Entry_Date', 'Close': 'Entry_Price'}, inplace=True)
entries['Exit_Date'] = entries['Entry_Date'] + pd.Timedelta(days=1)

# Merge with exit prices
exits = data[['Date', 'Close']]
exits.rename(columns={'Date': 'Exit_Date', 'Close': 'Exit_Price'}, inplace=True)
trades = pd.merge(entries, exits, on='Exit_Date', how='left')

# Handle non-trading days
trades['Exit_Price'].fillna(method='bfill', inplace=True)

# Calculate returns
trades['P&L'] = trades['Exit_Price'] - trades['Entry_Price']
trades['Returns'] = trades['P&L'] / trades['Entry_Price']

# **Limit to first 230 trades**
trades = trades.head(230)

# Calculate equity curve
initial_capital = 100000
trades['Cumulative_Returns'] = (1 + trades['Returns']).cumprod()
trades['Equity'] = initial_capital * trades['Cumulative_Returns']
trades.reset_index(drop=True, inplace=True)
trades['Trade_Number'] = trades.index + 1

# Plot equity curve
plt.figure(figsize=(12,6))
plt.plot(trades['Trade_Number'], trades['Equity'], marker='o')
plt.title('Equity Curve of Turnaround Monday Strategy (First 230 Trades)')
plt.xlabel('Trade Number')
plt.ylabel('Equity ($)')
plt.grid(True)
plt.show()

# Calculate statistical performance metrics

# Average Profit Per Trade
average_profit_per_trade = trades['P&L'].mean()
print(f"Average Profit Per Trade: ${average_profit_per_trade:.2f}")

# Total Number of Trades
total_number_of_trades = len(trades)
print(f"Total Number of Trades: {total_number_of_trades}")

# Total Profitable Trades and Percent Profitable Trades
profitable_trades = trades[trades['P&L'] > 0]
total_profitable_trades = len(profitable_trades)
percent_profitable_trades = (total_profitable_trades / total_number_of_trades) * 100
print(f"Total Profitable Trades: {total_profitable_trades}")
print(f"Percent Profitable Trades: {percent_profitable_trades:.2f}%")

# Highest Single Trade Profit
highest_single_trade_profit = trades['P&L'].max()
print(f"Highest Single Trade Profit: ${highest_single_trade_profit:.2f}")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create 1D arrays for entry_time and stop_loss (by repeating values)
entry_time = np.tile([9, 10, 11, 12, 13], 5)  # Repeats for each stop_loss
stop_loss = np.repeat([0.5, 1.0, 1.5, 2.0, 2.5], 5)  # Repeats for each entry_time

# Generate random Sharpe ratios for each combination of entry_time and stop_loss
sharpe_ratio = np.random.rand(25)  # 25 values corresponding to 5x5 grid

# Create a DataFrame with the correct 1D arrays
df = pd.DataFrame({
    'entry_time': entry_time,
    'stop_loss': stop_loss,
    'sharpe_ratio': sharpe_ratio
})

# Pivot the DataFrame to get a matrix format for heatmap
df_pivot = pd.pivot_table(df, values='sharpe_ratio', index='stop_loss', columns='entry_time')

# Set the size of the plot
plt.figure(figsize=(10, 8))

# Create the heatmap
sns.heatmap(df_pivot, annot=True, cmap='coolwarm', cbar_kws={'label': 'Sharpe Ratio'})

# Customize labels and title
plt.title('Turnaround Monday Strategy Optimization: Entry Time vs Stop Loss')
plt.xlabel('Entry Time (Hours)')
plt.ylabel('Stop Loss (%)')

# Show the plot
plt.show()

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to fetch 1-hour data
def fetch_hourly_data(symbol, period='2y'):
    data = yf.download(symbol, period=period, interval='1h')
    data.dropna(inplace=True)
    return data

# Backtesting function
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

        trades_df, performance = backtest_strategy(data)
        performance['Symbol'] = symbol
        all_performance.append(performance)

        if not trades_df.empty:
            all_trades.append(trades_df)

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

# Compile performance summary
performance_df = pd.DataFrame(all_performance)
performance_df = performance_df[['Symbol', 'Total Trades', 'Total Profitable Trades',
                                 'Percent Profitable Trades', 'Average Profit Per Trade',
                                 'Highest Single Trade Profit']]

print("\nBacktesting Performance Summary:")
print(performance_df)

# concatenate all trades
if all_trades:
    all_trades_df = pd.concat(all_trades, ignore_index=True)

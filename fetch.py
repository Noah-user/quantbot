import yfinance as yf
from datetime import datetime, timedelta
import os

# Save data locally and provide feedback
def save_to_csv(dataframe, filepath, interval):
    try:
        dataframe.to_csv(filepath, index=False)
        print(f"Successfully saved {interval} data to {filepath}")
    except Exception as e:
        print(f"Failed to save {interval} data to {filepath}. Error: {e}")

# Function to fetch and save data
def fetch_data(folder_name, ticker="NQ=F"):
    # Calculate dynamic dates
    end_date = datetime.today().strftime('%Y-%m-%d')  # Today's date

    # Create the folder for saving CSV files
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  # Create folder if it doesn't exist

    # Change start date for daily data
    start_date = (datetime.today() - timedelta(days=3000)).strftime('%Y-%m-%d')  # 3000 days
    data_1d = yf.download(ticker, interval="1d", start=start_date, end=end_date)

    save_to_csv(data_1d, os.path.join(folder_name, "nq_data_1d.csv"), "1-day")
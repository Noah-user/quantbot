from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import ta
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import StochasticOscillator, ROCIndicator

def load_and_prepare_data(csv_path):
    """
    Load the data and prepare it for training and testing.
    """
    # Load data, skipping the second row
    data = pd.read_csv(csv_path, skiprows=[1])
    
    # Ensure all necessary columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Missing required columns in the data.")
    
    # Convert columns to numeric
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)  # Drop rows with missing values
    
    # Apply log transformation to Volume
    data['Volume'] = np.log1p(data['Volume'])  # log(1 + x)
    
    # Add technical indicators
    data['SMA_10'] = data['Close'].rolling(window=10).mean()  # 10-day Simple Moving Average
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()  # 14-day RSI
    
    # MACD (Moving Average Convergence Divergence)
    macd_indicator = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd_indicator.macd()  # MACD line
    data['Signal_Line'] = macd_indicator.macd_signal()  # Signal Line
    data['MACD_Histogram'] = macd_indicator.macd_diff()  # MACD Histogram
    
    # Bollinger Bands
    bb_indicator = BollingerBands(data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb_indicator.bollinger_hband()  # Upper Bollinger Band
    data['BB_Low'] = bb_indicator.bollinger_lband()   # Lower Bollinger Band
    
    # Average True Range (ATR)
    atr_indicator = AverageTrueRange(data['High'], data['Low'], data['Close'], window=14)
    data['ATR'] = atr_indicator.average_true_range()
    
    # Stochastic Oscillator
    stoch_indicator = StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
    data['Stochastic'] = stoch_indicator.stoch()
    
    # Rate of Change (ROC)
    roc_indicator = ROCIndicator(data['Close'], window=14)
    data['ROC'] = roc_indicator.roc()
    
    # Price differences and returns
    data['Close_Diff'] = data['Close'].diff()  # Daily price change
    data['Close_Return'] = data['Close'].pct_change()  # Daily percentage return
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))  # Log returns
    
    # Add sequential features
    data['Sequence'] = np.arange(len(data))  # Sequential index (0, 1, 2, ...)
    
    # Cyclic encoding of the sequence
    max_sequence = len(data)
    data['Sequence_Sin'] = np.sin(2 * np.pi * data['Sequence'] / max_sequence)  # Sine transformation
    data['Sequence_Cos'] = np.cos(2 * np.pi * data['Sequence'] / max_sequence)  # Cosine transformation
    
    # VWAP (Volume-Weighted Average Price)
    data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3  # Typical price
    data['Price_Volume'] = data['Typical_Price'] * data['Volume']  # Price multiplied by volume
    data['Cumulative_Price_Volume'] = data['Price_Volume'].cumsum()  # Cumulative sum of price * volume
    data['Cumulative_Volume'] = data['Volume'].cumsum()  # Cumulative sum of volume
    data['VWAP'] = data['Cumulative_Price_Volume'] / data['Cumulative_Volume']  # VWAP formula
    
    # Drop intermediate columns used for VWAP calculation
    data.drop(columns=['Typical_Price', 'Price_Volume', 'Cumulative_Price_Volume', 'Cumulative_Volume'], inplace=True)
    
    # Drop rows with NaN values caused by rolling/indicator calculations
    data.dropna(inplace=True)
    
    # Define features (X) and target (y)
    X = data[['Open', 'High', 'Low', 'Close', 'Volume',
              'SMA_10', 'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram',
              'BB_High', 'BB_Low', 'ATR', 'Stochastic', 'ROC',
              'Close_Diff', 'Close_Return', 'Log_Return',
              'Sequence', 'Sequence_Sin', 'Sequence_Cos', 'VWAP']]  # Include Sequence and cyclic features
    
    y = data[['Open', 'High', 'Low', 'Close',]].shift(-1)  # Predict next day's OHLC
    X = X[:-1]  # Remove the last row to match y's length
    y = y[:-1]  # Remove the NaN target value
    
    # Normalize each feature separately
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    return X_scaled, y_scaled, X_scaler, y_scaler, data[:-1]  # Return preprocessed data without the last row


def build_and_train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=16):
    """
    Build and train the LSTM model with early stopping.
    """
    # Define model
    model_bilstm = Sequential([
    Bidirectional(LSTM(256, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
    Dropout(0.3),
    Bidirectional(LSTM(128, activation='relu', return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64, activation='relu', return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(4)  # Output layer for 4 predictions
])
    
    # Compile the model
    model_bilstm.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=10,         # Stop if no improvement for 10 consecutive epochs
        restore_best_weights=True  # Restore weights from the best epoch
    )
    
    # Train the model
    history = model_bilstm.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model_bilstm, history


def predict_all(model, X, y_scaler):
    """
    Predict all next-day values using the trained model.
    Scale the predictions back to the original range.
    """
    # Debugging: Print the shape of X before prediction
    print(f"Shape of X before prediction: {X.shape}")
    
    # Predict in normalized space
    predictions = model.predict(X)
    
    # Debugging: Print the shape of predictions
    print(f"Shape of predictions: {predictions.shape}")
    
    # Inverse transform predictions to the original range
    predictions_original_scale = y_scaler.inverse_transform(predictions)
    
    # No need to reverse-transform Volume since it's excluded
    return predictions_original_scale
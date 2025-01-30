from fetch import fetch_data
from predict import load_and_prepare_data, build_and_train_model, predict_all
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Define the folder path and ticker
folder_name = r"YOUR-FOLDER-NAME"
ticker = "NQ=F"  # Nasdaq-100 Futures
csv_path = r"YOUR-CSV-PATH"

# Call the fetch function
fetch_data(folder_name, ticker)

# Load and prepare data
X, y, X_scaler, y_scaler, data = load_and_prepare_data(csv_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape X_train and X_test for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Add a time step dimension
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build and train the model
model_bilstm, history = build_and_train_model(X_train, y_train, X_test, y_test)

# Predict all next-day values (for visualization)
full_predictions = predict_all(model_bilstm, X_test, y_scaler)  # No redundant reshape

# Assign predictions to the DataFrame
data[['Predicted_Open', 'Predicted_High', 'Predicted_Low', 'Predicted_Close']] = pd.DataFrame(
    full_predictions, index=data.index[-len(full_predictions):]
)

# Shift predictions by one row to align with the next day's values
data[['Predicted_Open', 'Predicted_High', 'Predicted_Low', 'Predicted_Close']] = data[
    ['Predicted_Open', 'Predicted_High', 'Predicted_Low', 'Predicted_Close']
].shift(1)

# Drop rows with NaN predictions
data.dropna(inplace=True)

# Print the last 10 rows (most recent predictions)
print("\nLast 10 Rows (Most Recent Predictions):")
print(data[['Open', 'Predicted_Open']].tail(10))
print(data[['Close', 'Predicted_Close']].tail(10))
print(data[['High', 'Predicted_High']].tail(10))
print(data[['Low', 'Predicted_Low']].tail(10))

# Evaluate the model on the test set
test_predictions = model_bilstm.predict(X_test)

# Debugging: Print the shape of test_predictions
print(f"Shape of test_predictions: {test_predictions.shape}")

# Ensure shapes match
assert y_test.shape == test_predictions.shape, f"Shape mismatch: y_test={y_test.shape}, test_predictions={test_predictions.shape}"

# Rescale test predictions and y_test back to original scale
y_test_rescaled = y_scaler.inverse_transform(y_test)
test_predictions_rescaled = y_scaler.inverse_transform(test_predictions)

# Calculate MSE and MAE for each column
mae_values = {}
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    mse_scaled = mean_squared_error(y_test[:, i], test_predictions[:, i])
    mae_original = mean_absolute_error(y_test_rescaled[:, i], test_predictions_rescaled[:, i])
    mae_values[col] = mae_original
    print(f"MSE for {col} (scaled): {mse_scaled:.6f}")
    print(f"MAE for {col} (original scale): {mae_original:.2f}")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot actual vs. predicted values
plt.figure(figsize=(14, 8))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2, 3, i + 1)
    plt.plot(y_test_rescaled[:, i], label=f'Actual {col}', color='blue')
    plt.plot(test_predictions_rescaled[:, i], label=f'Predicted {col}', linestyle='--', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'{col} Prediction')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

# Extract the next day's prediction (last row of test predictions)
next_day_prediction = test_predictions_rescaled[-1]

# Print the next day's predicted values
print("\nNext Day's Predicted Values:")
print(f"Open: {next_day_prediction[0]:.2f}")
print(f"High: {next_day_prediction[1]:.2f}")
print(f"Low: {next_day_prediction[2]:.2f}")
print(f"Close: {next_day_prediction[3]:.2f}")
percent_open = ((mae_values['Open'] / next_day_prediction[0]) * 100)
print(f"Percent off open: {percent_open:.2f}")
percent_high = ((mae_values['High'] / next_day_prediction[1]) * 100)
print(f"Percent off high: {percent_high:.2f}")
percent_low = ((mae_values['Low'] / next_day_prediction[2]) * 100)
print(f"Percent off low: {percent_low:.2f}")
percent_close = ((mae_values['Close'] / next_day_prediction[3]) * 100)
print(f"Percent off close: {percent_close:.2f}")
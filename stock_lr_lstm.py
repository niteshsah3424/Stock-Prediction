import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

# Load CSV
df = pd.read_csv("stock_data.csv")

# IMPORTANT FIX: Select ONLY numeric Close column
df = df[['Close']]

# Convert Close to numeric (force remove text like AAPL)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# Remove rows with NaN
df.dropna(inplace=True)

# Scaling
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df[['Close']])

# Prepare data
X = np.arange(len(df)).reshape(-1, 1)
y = df['Close'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

pred = lr.predict(X_test)

rmse = math.sqrt(mean_squared_error(y_test, pred))
print("Linear Regression RMSE:", rmse)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

dataset = df[['Close']].values
X_lstm, y_lstm = create_dataset(dataset)

X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

train_size = int(len(X_lstm) * 0.8)

X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32)

lstm_pred = model.predict(X_test)
lstm_rmse = math.sqrt(mean_squared_error(y_test, lstm_pred))

print("LSTM RMSE:", lstm_rmse)

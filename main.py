import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tkinter as tk

# Налаштування Binance API
API_KEY = "ваш_api_key"
API_SECRET = "ваш_api_secret"
client = Client(API_KEY, API_SECRET)

# Отримання даних
def get_historical_data():
    klines = client.get_historical_klines("PEPEUSDT", Client.KLINE_INTERVAL_1HOUR, "24 hours ago UTC")
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 
                                         'volume', 'close_time', 'quote_asset_volume', 
                                         'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    data['close'] = data['close'].astype(float)
    return data['close'].values

# Підготовка послідовностей для LSTM
def prepare_data(data, look_back=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:i + look_back, 0])
        y.append(data_scaled[i + look_back, 0])
    return np.array(X), np.array(y), scaler

# Створення та навчання моделі LSTM
def create_lstm_model(look_back=5):
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1), return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Прогнозування ціни
def predict_price():
    data = get_historical_data()
    look_back = 5
    X, y, scaler = prepare_data(data, look_back)
    
    # Перетворення даних для LSTM (додавання розмірності)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Навчання моделі
    model = create_lstm_model(look_back)
    model.fit(X, y, epochs=10, batch_size=1, verbose=0)
    
    # Прогноз на наступну годину
    last_sequence = data[-look_back:]
    last_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    last_sequence = np.reshape(last_sequence, (1, look_back, 1))
    predicted_scaled = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
    
    return predicted_price

# Оновлення інтерфейсу
def update_price_label():
    try:
        predicted_price = predict_price()
        label.config(text=f"Через годину вартість буде такою: ${predicted_price:.8f}")
    except Exception as e:
        label.config(text=f"Помилка: {str(e)}")
    root.after(300000, update_price_label)  # Оновлення кожні 5 хвилин

# Створення вікна
root = tk.Tk()
root.title("Прогноз ціни PEPE (LSTM)")
root.geometry("400x200")

label = tk.Label(root, text="Завантаження...", font=("Arial", 14))
label.pack(expand=True)

update_price_label()
root.mainloop()

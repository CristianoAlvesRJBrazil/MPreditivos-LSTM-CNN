import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping

class BitcoinPredictor:
    def __init__(self, ticker='BTC-USD', start='2020-01-01', end='2024-10-01', time_step=5, prediction_steps=1, train_ratio=0.8):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.time_step = time_step
        self.prediction_steps = max(1, prediction_steps)
        self.train_ratio = train_ratio
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        self.train_data = None
        self.test_data = None
        self.predictions = None
        self.real_values = None

    def download_data(self):
        try:
            df = yf.download(self.ticker, start=self.start, end=self.end)
            df['Close'] = df['Close'].astype(float)
            self.data = df['Close'].values.reshape(-1, 1)
            print("Dados baixados com sucesso.")
            return df
        except Exception as e:
            print("Erro ao baixar dados:", e)
            return None

    def preprocess_data(self):
        scaled_data = self.scaler.fit_transform(self.data)
        train_size = int(len(scaled_data) * self.train_ratio)
        self.train_data = scaled_data[:train_size]
        self.test_data = scaled_data[train_size:]
        print("Dados pré-processados.")

    def create_dataset(self, data):
        X, Y = [], []
        for i in range(len(data) - self.time_step - self.prediction_steps + 1):
            X.append(data[i:(i + self.time_step), 0])
            Y.append(data[(i + self.time_step):(i + self.time_step + self.prediction_steps), 0])
        return np.array(X), np.array(Y)

    def build_bidirectional_model(self):
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=(self.time_step, 1)),
            Dropout(0.3),
            Bidirectional(LSTM(100, return_sequences=False)),
            Dropout(0.3),
            Dense(100, activation='relu'),
            Dense(self.prediction_steps)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        print("Modelo bidirecional LSTM criado.")

    def train_model(self, batch_size=32, patience=5):
        X_train, y_train = self.create_dataset(self.train_data)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        early_stop = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True, verbose=1)
        self.model.fit(X_train, y_train, epochs=100, batch_size=batch_size, callbacks=[early_stop], verbose=1)
        print("Treinamento concluído com early stopping.")

    def predict(self):
        X_test, y_test = self.create_dataset(self.test_data)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        predictions = self.model.predict(X_test)
        self.predictions = self.scaler.inverse_transform(predictions)
        self.real_values = self.scaler.inverse_transform(y_test)
        return self.predictions

    def predict_new_data(self, recent_data):
        recent_data_scaled = self.scaler.transform(recent_data)
        X_new = [recent_data_scaled[-self.time_step:, 0]]
        X_new = np.array(X_new).reshape(1, self.time_step, 1)
        
        predicted_scaled = self.model.predict(X_new)
        predicted = self.scaler.inverse_transform(predicted_scaled)
        return predicted

    def plot_results(self, df, predictions, future_price):
        train_size = int(len(self.data) * self.train_ratio)
        valid = df[train_size + self.time_step:train_size + self.time_step + len(predictions)]
        valid['Predictions'] = [pred[0] for pred in predictions]
        
        plt.figure(figsize=(16, 8))
        plt.title('Modelo de Previsão de Preços do Bitcoin')
        plt.xlabel('Data')
        plt.ylabel('Preço do Bitcoin (USD)')
        plt.plot(df['Close'][:train_size + self.time_step], label='Treinamento')
        plt.plot(valid['Close'], label='Valores Reais')
        plt.plot(valid['Predictions'], label='Previsão')
        
        plt.axhline(y=future_price, color='blue', linestyle='--', label='Preço Futuro Previsto')
        
        plt.legend(loc='lower right')
        plt.show()

if __name__ == "__main__":
    btc_predictor = BitcoinPredictor()
    df = btc_predictor.download_data()
    
    if df is not None:  
        btc_predictor.preprocess_data()
        btc_predictor.build_bidirectional_model()
        btc_predictor.train_model(batch_size=32, patience=5)
        
        predictions = btc_predictor.predict()
        
        recent_data = df['Close'][-btc_predictor.time_step:].values.reshape(-1, 1)
        predicted_price = btc_predictor.predict_new_data(recent_data)[0, 0]
        
        btc_predictor.plot_results(df, predictions, predicted_price)

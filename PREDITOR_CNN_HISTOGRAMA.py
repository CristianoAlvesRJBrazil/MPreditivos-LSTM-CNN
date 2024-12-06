import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from scipy.stats import norm
import seaborn as sns
import csv
from scipy.stats import norm, mode  # Adicionando a importação da função mode


class BitcoinPredictor:
    def __init__(self, ticker='BTC-USD', start='2020-01-01', end='2024-11-01', time_step=5, prediction_steps=1, train_ratio=0.8):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.time_step = time_step
        self.prediction_steps = max(1, prediction_steps)  # Garantir que prediction_steps >= 1
        self.train_ratio = train_ratio
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        self.predictions = None
        self.real_values = None

    def download_data(self):
        """Baixa os dados históricos do Bitcoin e armazena em formato 'Close'."""
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
        """Escalona os dados e separa em conjuntos de treino e teste."""
        scaled_data = self.scaler.fit_transform(self.data)
        train_size = int(len(scaled_data) * self.train_ratio)
        self.train_data = scaled_data[:train_size]
        self.test_data = scaled_data[train_size:]
        print("Dados pré-processados.")

    def create_dataset(self, data):
        """
        Gera o conjunto de dados com base em time_step e prediction_steps.
        
        Parâmetros:
            data (array): Conjunto de dados escalonado para gerar amostras.
        
        Retorna:
            X, Y (arrays): Dados de entrada e saída para treinamento/teste.
        """
        X, Y = [], []
        for i in range(len(data) - self.time_step - self.prediction_steps + 1):
            X.append(data[i:(i + self.time_step), 0])
            Y.append(data[(i + self.time_step):(i + self.time_step + self.prediction_steps), 0])
        return np.array(X), np.array(Y)

    def build_model(self):
        """Cria o modelo híbrido Bidirectional LSTM + CNN para previsão."""
        model = Sequential([
            # Camada de Convolução 1D para capturar padrões locais
            Conv1D(256, 2, activation='relu', input_shape=(self.time_step, 1)),
            MaxPooling1D(2),
            Dropout(0.2),
            
            # Camadas LSTM Bidirecionais para capturar dependências temporais
            Bidirectional(LSTM(100, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(100, return_sequences=False)),
            Dropout(0.3),
            
            # Camadas densas para ajustar a saída
            Dense(100, activation='relu'),
            Dense(self.prediction_steps)  # Saída para múltiplos dias
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        print("Modelo híbrido Bidirectional LSTM + CNN criado.")

    def train_model(self, batch_size=1, patience=5):
        """Treina o modelo com early stopping para evitar overfitting."""
        X_train, y_train = self.create_dataset(self.train_data)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        
        early_stop = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True, verbose=1)
        self.model.fit(X_train, y_train, epochs=100, batch_size=batch_size, callbacks=[early_stop], verbose=1)
        print("Treinamento concluído com early stopping.")

    def predict(self):
        """Gera previsões usando o conjunto de dados de teste e desfaz a escala para os valores reais."""
        X_test, y_test = self.create_dataset(self.test_data)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        predictions = self.model.predict(X_test)
        self.predictions = self.scaler.inverse_transform(predictions)
        self.real_values = self.scaler.inverse_transform(y_test)
        return self.predictions

    def save_predictions_to_csv(self, df, filename="bitcoin_predictions_lstm_cnn.csv"):
        """
        Salva previsões e valores reais em CSV para análise futura.
        
        Parâmetros:
            df (DataFrame): Dados históricos do Bitcoin.
            filename (str): Nome do arquivo CSV (padrão: 'bitcoin_predictions.csv').
        """
        train_size = int(len(self.data) * self.train_ratio)
        dates = df.index[train_size + self.time_step:train_size + self.time_step + len(self.predictions)]
        
        predictions_df = pd.DataFrame({
            'Date': dates,
            'Real_Close': [real[0] for real in self.real_values],
            'Predicted_Close': [pred[0] for pred in self.predictions]
        })
        
        predictions_df.to_csv(filename, index=False)
        print(f"Previsões e valores reais salvos em {filename}")

    def plot_results(self, df, predictions):
        """Plota os resultados de previsões vs. valores reais."""
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
        plt.legend(loc='lower right')
        plt.show()

# Execução principal
if __name__ == "__main__":
    best_losses = []  # Lista para armazenar os melhores valores de loss
    n_runs = 100  # Número de execuções
    
    for run in range(n_runs):
        print(f"Execução {run + 1}/{n_runs}")
        btc_predictor = BitcoinPredictor()
        df = btc_predictor.download_data()
        
        if df is not None:
            btc_predictor.preprocess_data()
            btc_predictor.build_model()
            
            # Configura o treinamento com early stopping
            X_train, y_train = btc_predictor.create_dataset(btc_predictor.train_data)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)
            
            history = btc_predictor.model.fit(
                X_train, y_train, 
                epochs=100, 
                batch_size=32, 
                callbacks=[early_stop], 
                verbose=1
            )
            
            # Captura o menor valor de loss
            best_loss = min(history.history['loss'])
            best_losses.append(best_loss)
            print(f"Melhor Loss da execução {run + 1}: {best_loss}")
    
    # Salvar os melhores loss em CSV
    with open("best_losses_CNN.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Execution", "Best Loss"])
        for i, loss in enumerate(best_losses):
            writer.writerow([i + 1, loss])
    
    # Estatísticas descritivas
    mean_loss = np.mean(best_losses)
    std_loss = np.std(best_losses)
    min_loss = np.min(best_losses)
    max_loss = np.max(best_losses)
    median_loss = np.median(best_losses)
    
    # Calcular a moda
    mode_result = mode(best_losses, nan_policy="omit")
    
    # Verificar se existe mais de uma moda e tratar
    if isinstance(mode_result.mode, np.ndarray):
        mode_loss = mode_result.mode[0]  # Caso seja um array, pegamos o primeiro valor
    else:
        mode_loss = mode_result.mode  # Caso seja um valor único, usamos diretamente
    
    # Salvar as estatísticas descritivas em um arquivo CSV
    with open("loss_statistics_CNN.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Mean", mean_loss])
        writer.writerow(["Standard Deviation", std_loss])
        writer.writerow(["Minimum", min_loss])
        writer.writerow(["Maximum", max_loss])
        writer.writerow(["Median", median_loss])
        writer.writerow(["Mode", mode_loss])  # Salva a moda
    
    # Gerar o histograma com a curva normal
    plt.figure(figsize=(10, 6))
    sns.histplot(best_losses, bins=10, kde=False, color='skyblue', label="Frequência", stat="density", edgecolor="black")
    
    # Ajustar a curva normal
    x = np.linspace(min_loss, max_loss, 100)
    y = norm.pdf(x, mean_loss, std_loss)
    plt.plot(x, y, color='red', label='Curva Normal')
    
    # Adicionar informações ao gráfico
    plt.title("Distribuição dos Melhores Loss - CNN_LSTM")
    plt.xlabel("Loss")
    plt.ylabel("Densidade")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("Melhor_MSE_Curva_Normal_CNN.png")  # Salva o gráfico
    plt.show()

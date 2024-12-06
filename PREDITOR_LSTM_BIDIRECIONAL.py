import yfinance as yf
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping


# Classe para previsão de preços do Bitcoin
class BitcoinPredictor:
    def __init__(self, ticker='BTC-USD', start='2020-01-01', end='2024-11-01', time_step=5, prediction_steps=1, train_ratio=0.8):
        """
        Inicializa o preditor de preços do Bitcoin.
        """
        self.ticker = ticker  # Código do ativo financeiro no Yahoo Finance
        self.start = start  # Data de início para os dados
        self.end = end  # Data de término para os dados
        self.time_step = time_step  # Número de passos de tempo usados como entrada
        self.prediction_steps = max(1, prediction_steps)  # Número de passos a prever
        self.train_ratio = train_ratio  # Proporção dos dados para treinamento
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Normalizador dos dados
        self.model = None  # Modelo LSTM
        self.data = None  # Dados brutos
        self.train_data = None  # Dados de treinamento
        self.test_data = None  # Dados de teste
        self.predictions = None  # Previsões do modelo
        self.real_values = None  # Valores reais

    def download_data(self):
        """
        Baixa os dados históricos do ativo usando o Yahoo Finance.
        """
        try:
            df = yf.download(self.ticker, start=self.start, end=self.end)
            df['Close'] = df['Close'].astype(float)  # Usa preços de fechamento
            self.data = df['Close'].values.reshape(-1, 1)
            print("Dados baixados com sucesso.")
            return df
        except Exception as e:
            print("Erro ao baixar dados:", e)
            return None

    def preprocess_data(self):
        """
        Pré-processa os dados, incluindo normalização e divisão em treino e teste.
        """
        scaled_data = self.scaler.fit_transform(self.data)  # Normaliza os dados
        train_size = int(len(scaled_data) * self.train_ratio)  # Determina o tamanho do conjunto de treino
        self.train_data = scaled_data[:train_size]  # Dados de treino
        self.test_data = scaled_data[train_size:]  # Dados de teste
        print("Dados pré-processados.")

    def create_dataset(self, data):
        """
        Cria conjuntos de entrada (X) e saída (Y) a partir dos dados.
        """
        X, Y = [], []
        for i in range(len(data) - self.time_step - self.prediction_steps + 1):
            X.append(data[i:(i + self.time_step), 0])  # Dados de entrada
            Y.append(data[(i + self.time_step):(i + self.time_step + self.prediction_steps), 0])  # Dados de saída
        return np.array(X), np.array(Y)

    def build_bidirectional_model(self):
        """
        Cria um modelo de rede neural LSTM bidirecional.
        """
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=(self.time_step, 1)),  # Primeira camada LSTM bidirecional
            Dropout(0.3),  # Regularização para evitar overfitting
            Bidirectional(LSTM(100, return_sequences=False)),  # Segunda camada LSTM bidirecional
            Dropout(0.3),
            Dense(100, activation='relu'),  # Camada densa intermediária
            Dense(self.prediction_steps)  # Camada de saída
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')  # Compila o modelo com perda MSE
        self.model = model
        print("Modelo bidirecional LSTM criado.")

    def train_model(self, batch_size=32, patience=5):
        """
        Treina o modelo utilizando os dados de treinamento.
        """
        X_train, y_train = self.create_dataset(self.train_data)  # Cria conjunto de treinamento
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Ajusta as dimensões

        # Configura o early stopping para parar o treinamento se a perda não melhorar
        early_stop = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True, verbose=1)
        
        # Treina o modelo
        self.model.fit(X_train, y_train, epochs=100, batch_size=batch_size, callbacks=[early_stop], verbose=1)
        print("Treinamento concluído com early stopping.")

    def predict(self):
        """
        Faz previsões utilizando os dados de teste.
        """
        X_test, y_test = self.create_dataset(self.test_data)  # Cria conjunto de teste
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # Ajusta as dimensões
        predictions = self.model.predict(X_test)  # Faz previsões
        self.predictions = self.scaler.inverse_transform(predictions)  # Desnormaliza previsões
        self.real_values = self.scaler.inverse_transform(y_test)  # Desnormaliza valores reais
        return self.predictions

    def predict_new_data(self, recent_data):
        """
        Faz previsões baseadas em novos dados recentes.
        """
        recent_data_scaled = self.scaler.transform(recent_data)  # Normaliza os dados recentes
        X_new = [recent_data_scaled[-self.time_step:, 0]]  # Usa apenas os últimos time_step valores
        X_new = np.array(X_new).reshape(1, self.time_step, 1)  # Ajusta as dimensões
        
        predicted_scaled = self.model.predict(X_new)  # Faz a previsão
        predicted = self.scaler.inverse_transform(predicted_scaled)  # Desnormaliza o resultado
        return predicted

    def plot_results(self, df, predictions, future_price):
        """
        Plota os resultados das previsões e os valores reais.
        """
        train_size = int(len(self.data) * self.train_ratio)
        valid = df[train_size + self.time_step:train_size + self.time_step + len(predictions)]
        valid['Predictions'] = [pred[0] for pred in predictions]  # Adiciona previsões ao dataframe
        
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
    
    def save_predictions_to_csv(self, df, file_name='bitcoin_predictions_lstm.csv'):
        if self.real_values is not None and self.predictions is not None:
            # Obter tamanho do conjunto de teste
            train_size = int(len(self.data) * self.train_ratio)
            test_dates = df.index[train_size + self.time_step:train_size + self.time_step + len(self.predictions)]
            
            # Certificar-se de que os arrays possuem o mesmo tamanho
            real_values = self.real_values[:len(self.predictions)]
            
            # Criar um DataFrame com as datas, valores reais e previstos
            data = {
                'Data': test_dates,
                'Real': [val[0] for val in real_values],
                'Previsto': [val[0] for val in self.predictions]
            }
            result_df = pd.DataFrame(data)
            
            # Salvar o DataFrame como CSV
            result_df.to_csv(file_name, index=False)
            print(f"Previsões salvas no arquivo {file_name}.")
        else:
            print("Nenhuma previsão encontrada para salvar.")

# Execução principal
if __name__ == "__main__":
    # Cria a instância do preditor
    btc_predictor = BitcoinPredictor()
    
    # Baixa os dados
    df = btc_predictor.download_data()
    
    if df is not None:  
        btc_predictor.preprocess_data() # Pré-processa os dados
        btc_predictor.build_bidirectional_model() # Cria o modelo
        btc_predictor.train_model(batch_size=32, patience=5) # Treina o modelo
        
        predictions = btc_predictor.predict() # Faz previsões
        
        # Salvar previsões com as datas em um CSV
        btc_predictor.save_predictions_to_csv(df, 'bitcoin_predictions_lstm.csv')
        
        # Faz previsão baseada em dados recentes
        recent_data = df['Close'][-btc_predictor.time_step:].values.reshape(-1, 1)
        predicted_price = btc_predictor.predict_new_data(recent_data)[0, 0]
        
        # Plota os resultados
        btc_predictor.plot_results(df, predictions, predicted_price)




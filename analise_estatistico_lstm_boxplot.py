import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Substitua 'seu_arquivo.csv' pelo nome do seu arquivo CSV
arquivo_csv = 'bitcoin_predictions_lstm.csv'

# Carrega o dataset
df = pd.read_csv(arquivo_csv)

# Exibe as primeiras linhas do dataset
print(df.head(10))

# Calcula o erro relativo para cada linha
df['Erro_relativo'] = abs(df['Real'] - df['Previsto']) / abs(df['Real'])

# Exibe o resultado
print(df[['Erro_relativo']])

# Gráfico 1: Histograma com curva normal
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df['Erro_relativo'], bins=30, edgecolor='black', density=True, alpha=0.6, color='skyblue')

# Calcula estatísticas do erro relativo
media_erro_relativo = df['Erro_relativo'].mean()
desvio_padrao_erro_relativo = df['Erro_relativo'].std()

# Gera os valores para a curva normal
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, media_erro_relativo, desvio_padrao_erro_relativo)

# Plota a curva normal
plt.plot(x, p, 'r', linewidth=2)
title = f'Histograma e Curva Normal do Erro Relativo - LSTM\nMédia: {media_erro_relativo:.4f}, Desvio Padrão: {desvio_padrao_erro_relativo:.4f}'
plt.title(title)
plt.xlabel('Erro Relativo')
plt.ylabel('Densidade de Probabilidade')

# Exibe o histograma
plt.show()

# Gráfico 2: Boxplot do erro relativo
plt.figure(figsize=(8, 6))
plt.boxplot(df['Erro_relativo'], patch_artist=True, 
            boxprops=dict(facecolor='skyblue', color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='red'))
plt.title('Boxplot do Erro Relativo - LSTM')
plt.ylabel('Erro Relativo')

# Exibe o boxplot
plt.show()

# Imprime estatísticas básicas do erro relativo
print(f"Média do Erro relativo: {media_erro_relativo}")
print(f"Mediana do Erro relativo: {df['Erro_relativo'].median()}")
print(f"Desvio padrão do Erro relativo: {desvio_padrao_erro_relativo}")
print(df['Erro_relativo'].describe())

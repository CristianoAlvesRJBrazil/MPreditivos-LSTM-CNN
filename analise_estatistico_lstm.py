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
#df['erro_relativo'] = abs(df['Predicted_Close'] - df['Real_Close']) / abs(df['Real_Close'])
df['erro_relativo'] = abs(df['Real'] - df['Previsto'])/ abs(df['Real'])

# Exibe o resultado
print(df[['erro_relativo']])

# Configurações do histograma
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(df['erro_relativo'], bins=30, edgecolor='black', density=True, alpha=0.6, color='skyblue')

# Calcula estatísticas do erro relativo
media_erro_relativo = df['erro_relativo'].mean()
desvio_padrao_erro_relativo = df['erro_relativo'].std()

# Gera os valores para a curva normal
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, media_erro_relativo, desvio_padrao_erro_relativo)

# Plota a curva normal
plt.plot(x, p, 'r', linewidth=2)
title = f'Histograma e Curva Normal do Erro Relativo\nMédia: {media_erro_relativo:.4f}, Desvio Padrão: {desvio_padrao_erro_relativo:.4f}'
plt.title(title)
plt.xlabel('Erro Relativo')
plt.ylabel('Densidade de Probabilidade')

# Exibe o gráfico
plt.show()

# Imprime estatísticas básicas do erro relativo
print(f"Média do erro relativo: {media_erro_relativo}")
print(f"Mediana do erro relativo: {df['erro_relativo'].median()}")
print(f"Desvio padrão do erro relativo: {desvio_padrao_erro_relativo}")
print(df['erro_relativo'].describe())
 
 
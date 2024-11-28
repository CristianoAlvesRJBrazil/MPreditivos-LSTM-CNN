import pandas as pd
import numpy as np

# Carregar os dados do arquivo CSV
df_loss = pd.read_csv("bitcoin_predictions_lstm.csv")

# Garantir que as colunas 'Real' e 'Previsto' existam e não contenham valores nulos
if 'Real' in df_loss.columns and 'Previsto' in df_loss.columns:
    # Calcular a métrica 'Best Loss'
    df_loss['Best Loss'] = abs(df_loss['Real'] - df_loss['Previsto']) / abs(df_loss['Real'])
else:
    raise ValueError("As colunas 'Real' e 'Previsto' são obrigatórias no arquivo CSV.")

# Calcular as estatísticas descritivas
mean_loss = df_loss["Best Loss"].mean()  # Média
std_loss = df_loss["Best Loss"].std()    # Desvio padrão
min_loss = df_loss["Best Loss"].min()    # Valor mínimo
max_loss = df_loss["Best Loss"].max()    # Valor máximo
median_loss = df_loss["Best Loss"].median()  # Mediana

# Calcular a moda manualmente a partir da tabela de frequência
frequency_table = df_loss["Best Loss"].value_counts()
mode_loss = frequency_table.idxmax()  # Valor mais frequente
mode_count = frequency_table.max()    # Frequência da moda

# Criar um DataFrame para as estatísticas descritivas
data = {
    'Metric': ['Mean', 'Standard Deviation', 'Minimum', 'Maximum', 'Median', 'Mode'],
    'Value': [mean_loss, std_loss, min_loss, max_loss, median_loss, mode_loss]
}
df_stats = pd.DataFrame(data)

# Salvar as estatísticas descritivas em um arquivo CSV
output_file = "Estatistica_lstm_teste.csv"
df_stats.to_csv(output_file, index=False)
print(f"As estatísticas descritivas foram salvas em: {output_file}")

# Exibir a moda e sua frequência para validação adicional
print(f"Mode: {mode_loss}, Frequency: {mode_count}")

# Exibir o DataFrame com as estatísticas
df_stats

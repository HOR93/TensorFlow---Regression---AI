import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import math
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix

dropout_data = pd.read_csv("https://raw.githubusercontent.com/HOR93/csv_dataset/main/regress%C3%A3o_dataset_final%20-%20P%C3%A1gina1.csv")
visual = pd.read_csv("https://raw.githubusercontent.com/HOR93/csv_dataset/main/regress%C3%A3o_dataset_final%20-%20P%C3%A1gina1.csv")

visual["Course_A.Dropout Rate"] = visual["Course"] + " - " + visual["A.Dropout Rate"]

cores = plt.cm.get_cmap('tab20', len(visual['Course'].unique()))

plt.figure(figsize=(12, 8))
for i, curso in enumerate(visual['Course'].unique()):
    curso_df = visual[visual['Course'] == curso]
    plt.scatter(curso_df["Course_A.Dropout Rate"], curso_df["A.Dropout Rate"], c=[cores(i)], label=curso, s=10)

plt.title("Relationship between Course and A.Dropout Rate")
plt.xlabel("Course_A.Dropout Rate")
plt.ylabel("A.Dropout Rate")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks([])
plt.show()

dropout_data = dropout_data.replace(',', '.', regex=True)

dropout_data.head()

dropout_data = dropout_data.drop("Course", axis=1)

dropout_data.head()

#Column_transformer para tentar adequar e preparar os dados das colunas de forma eficiente no treinamento tanto as caracteristicas numericas e categoricas
ct = make_column_transformer(
    (MinMaxScaler(), ["R.Y", "Members", "R.retention", "R.Graduates", "R.Dropout", "Retention Rate", "Cmlt Graduation", "Cmlt Dropout", "A.Graduation Rate", "A.Dropout Rate"]),
    (OneHotEncoder(handle_unknown="ignore"), ["Pandemic"])
)

# separando os conjuntos de dados X para os dados para usar as previsões e y para a previsão
X = dropout_data.drop("A.Dropout Rate", axis=1)
y = dropout_data["A.Dropout Rate"]

#20% para o teste e 80% para treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#passando para float32 e evitar problemas com numpy
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

'''Column R.Y: int64
Column Members: int64
Column R.retention: int64
Column R.Graduates: int64
Column R.Dropout: int64
Column Retention Rate: object
Column Cmlt Graduation: object
Column Cmlt Dropout: object
Column  A.Graduation Rate: object
Column Pandemic: int64'''
for column in X_train.columns:
    print(f"Column {column}: {X_train[column].dtype}") # teste para ver se foi tudo convertido e exemplo anterior

tf.random.set_seed(42)

# modelo com 3 camadas densas, 100 neuronios e regularização para evitar um problema de overfitting que foi detectado no treinamento, e o quanto controla a regularização, e uma cada de 10 e uma de saida.
modelo = tf.keras.Sequential([
  tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])

# MAE como função de perda e adam para ajustar a taxa de aprendizado
modelo.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['mae'])


historico = modelo.fit(X_train, y_train, epochs=200, verbose=1)


#avaliando o resultado do treinamento
modelo.evaluate(X_test, y_test)

#analise grafica de desempenho
pd.DataFrame(historico.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs");

# usando predict para calcular as previsões do x_test
prevendo = modelo.predict(X_test)

print(prevendo)

#aqui é avaliado o desempenho geral do modelo, quanto menor MAE e RMSE(erro em relação aos valores reais) melhor mostra o desempenho
loss, mae = modelo.evaluate(X_test, y_test)

print("MAE:", mae)

previsoes = modelo.predict(X_test)

rmse = math.sqrt(mean_squared_error(y_test, prevendo))

print("RMSE:", rmse)

sns.set_style("whitegrid")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, prevendo, alpha=0.7, color="b", label="Real Values vs. Predictions")
plt.xlabel("Real Values (y_test)")
plt.ylabel("Predictions (prevendo)")
plt.title("Real Values vs. Predictions")
plt.legend()

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, linestyle='--')

plt.show()

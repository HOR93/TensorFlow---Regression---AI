import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelBinarizer

caminho_do_arquivo = 'https://raw.githubusercontent.com/HOR93/csv_dataset/main/EN_trajetoria_educacao_superior_2018_2022_engenharias%20-%20Engenharias.csv'
dataset = pd.read_csv(caminho_do_arquivo)
visual = pd.read_csv(caminho_do_arquivo)

dataset.tail()

dataset.isna().sum()

# passando para inteiros
for coluna in dataset.columns:
    if dataset[coluna].dtype == 'object':
        dataset[coluna] = pd.factorize(dataset[coluna])[0]

dataset

dataset = dataset.dropna()


# começando a separar as variaveis
encoder = LabelBinarizer()
encoder.fit(dataset['A.Dropout Rate'])
dataset['A.Dropout Rate'] = y = encoder.transform(dataset['A.Dropout Rate'])

dataset

### verificar quais colunas são importantes para o gráfico



sns.pairplot(dataset[["Course", "Members R. retention", "R.Graduates", "R.Dropout", "Retention Rate", "Cmlt Graduation", "Cmlt Dropout","A.Graduation Rate","A.Dropout Rate","Pandemic"]], diag_kind="kde")

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_dataset


train_stats = train_dataset.describe()
train_stats.pop("A.Dropout Rate")
train_stats = train_stats.transpose()
train_stats

#conjuntos de treinamento
train_labels = train_dataset.pop('A.Dropout Rate')
test_labels = test_dataset.pop('A.Dropout Rate')


train_labels

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

model.summary()

print("NaN present in labels:", train_labels.isna().any())

print("NaN present in original training data:", train_dataset.isna().any().any())

#normed_train_data = normed_train_data.fillna(normed_train_data.mean())

#normed_train_data= normed_train_data.dropna(how='all')

print("NaN present in train_dataset:", train_dataset.isna().any().any())

print("NaN present in test_dataset:", test_dataset.isna().any().any())

train_dataset = train_dataset.dropna(how='all')

#test_dataset = test_dataset.drop(test_dataset.index.difference(train_dataset.index))

print("NaN present in train_dataset:", train_dataset.isna().any().any())

print("NaN present in test_dataset:", test_dataset.isna().any().any())

train_dataset


example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
example_result

# Mostra o progresso do treinamento imprimindo um único ponto para cada epoch completada
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 50

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=1,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [A.Dropout Rate]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$A.Dropout Rate^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f}".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [A.Dropout Rate]')
plt.ylabel('Predictions [A.Dropout Rate]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [A.Dropout Rate]")
_ = plt.ylabel("Count")

# previsões
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [A.Dropout Rate]')
plt.ylabel('Predictions [A.Dropout Rate]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.figure()
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [A.Dropout Rate]")
_ = plt.ylabel("Count")
plt.show()


correlation_matrix = normed_train_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(8, 8))
sns.regplot(x=test_labels, y=test_predictions, scatter_kws={'alpha':0.5})
plt.xlabel('True Values [A.Dropout Rate]')
plt.ylabel('Predictions [A.Dropout Rate]')
plt.title('Relationship between True Values and Predictions')
plt.show()

import keras
from tensorflow.keras.datasets import mnist
import numpy as np
import time
import os

# Caminho para o tensorboard log dir
path = os.path.join(os.curdir, "logs")

# Nome do modelo - para já data de execução
log_dir = os.path.join(path, time.strftime("run_%Y_%m_%d-%H_%M_%S"))

# Criar modelo
inputs = keras.Input(shape=(28*28,))
fc1 = keras.layers.Dense(30*30, activation='relu')(inputs)
d1 = keras.layers.Dropout(0.1)(fc1, training=True)
fc2 = keras.layers.Dense(40*40, activation='relu')(d1)
d2 = keras.layers.Dropout(0.1)(fc2, training=True)
fc3 = keras.layers.Dense(30*30, activation='relu')(d2)
outputs = keras.layers.Dense(10, activation='softmax')(fc3)
model = keras.Model(inputs, outputs)

# Print do modelo e adicionar callback do tensorboard
model.summary()
tb = keras.callbacks.TensorBoard(log_dir)

# Dividir data em treino e teste
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = train_data.reshape(-1, 28*28).astype("float32") / 255
test_data = test_data.reshape(-1, 28*28).astype("float32") / 255

# Compilar e treinar modelo
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_data, train_labels, validation_split=0.1, batch_size=64,
          epochs=10, verbose=2, callbacks=[tb])

# Fazer uma ssembly de modelos (MC Dropout)
evals = np.stack([model.evaluate(test_data, test_labels) for _ in range(100)])
losses, accs = evals[:, 0], evals[:, 1]
mean_loss, mean_acc = losses.mean(axis=0), accs.mean(axis=0)
std_loss, std_acc = losses.std(axis=0), accs.std(axis=0)
print(f"Losses: {losses}")
print(f"Mean loss: {mean_loss} | Std loss: {std_loss}")
print(f"Accuracies: {accs}")
print(f"Mean acc: {mean_acc} | Std acc: {std_acc}")

# Save the model
model.save("first_model.h5")

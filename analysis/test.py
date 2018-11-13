import numpy as np
import tensorflow as tf

from keras import Sequential
from keras.layers import Lambda,Dense

def top_k(input, k):
  return tf.nn.top_k(input, k=k, sorted=True).indices

model = Sequential()
model.add(Dense(32, input_shape=(10,)))
model.add(Lambda(top_k, input_shape=(10,), arguments={'k': 4}))
model.summary()


model.compile(loss='mse',
          optimizer='adam')
    
data = np.array([
  [0, 5, 2, 1, 3, 6, 1, 2, 7, 4],
  [2, 4, 3, 1, 2, 0, 1, 5, 2, 4],
  [8, 9, 1, 8, 3, 0, 1, 3, 2, 6],
])

print(model.predict(x=data))
import tensorflow as tf 
import numpy as np
from tensorflow import keras 
number = int(input("Enter the x value you want to predict using a neural network of 1 neuron > "))
model = tf.keras.Sequential([keras.layers.Dense(input_shape=[1], units=1)])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1, 2, 3, 4, 5], dtype=int)
ys = np.array([2, 4, 6, 8, 10], dtype=int)

model.fit(xs, ys, epochs=1000)
print(model.predict([number]))

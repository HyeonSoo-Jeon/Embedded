import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv('./TensorFlow/gpascore.csv')
data = data.dropna()

y_data = data['admit'].values
x_data = []

for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(np.array(x_data), np.array(y_data), epochs=10)

# Prediction
forecast = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(forecast)

# 전처리
# 모델 튜닝

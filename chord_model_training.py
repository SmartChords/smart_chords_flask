import numpy as np
import pandas as pd
import sklearn
import random
import time
from sklearn import preprocessing, model_selection
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from keras.utils.np_utils import to_categorical


# loading the dataset
dataset = pd.read_csv("chord_data/chord_data_v2.csv")
data = dataset.values
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
X_train = ordinal_encoder.transform(X_train)
X_test = ordinal_encoder.transform(X_test)

# ordinal encode target variable
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_train = np_utils.to_categorical(y_train)
y_test = label_encoder.transform(y_test)
y_test = np_utils.to_categorical(y_test)


def create_model():
  model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(6,), kernel_initializer='he_normal'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(30, activation = 'softmax')
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  return model

## define the model
model = create_model()
# fit on the training set
model.fit(X_train, y_train, epochs = 350, batch_size = 35, verbose=2)
# predict on test set
yhat = model.predict(X_test)

# evaluate predictions
# scores = model.evaluate(X_test, y_test)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


model.save("chord_model")

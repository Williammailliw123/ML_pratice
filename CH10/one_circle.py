import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import utils
# Setting random seeds to get reproducible results
np.random.seed(0)
import tensorflow as tf
tf.random.set_seed(1)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def main():
    data=pd.read_csv("./CH10/one_circle.csv",index_col=0)
    x=np.array(data[['x_1','x_2']])
    y=np.array(data['y']).astype(int)
    utils.plot_points(x,y)
    plt.show()
    categorized_y = np.array(to_categorical(y, 2))
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(2,)))
    model.add(Dropout(.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(2, activation='softmax'))

    # Compiling the model
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x, categorized_y, epochs=100, batch_size=10)
    utils.plot_model(x, y, model)


if __name__ == '__main__':
    main()


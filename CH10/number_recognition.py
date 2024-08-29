import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def contruct_model(layer,Dropout_rate):
    model=Sequential()
    model.add(Dense(128,activation='relu',input_shape=(28*28,)))
    model.add(Dropout(Dropout_rate))
    for i in range(layer-1):
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(Dropout_rate))
    model.add(Dense(10,activation='softmax'))
    return model

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train_reshaped = x_train.reshape(-1,28*28)
    x_test_reshaped = x_test.reshape(-1,28*28)
    y_train_cat = to_categorical(y_train,10)
    y_test_cat = to_categorical(y_test,10)
    model=contruct_model(4,0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_reshaped,y_train_cat,epochs=20,batch_size=10)

    predictions_vector=model.predict(x_test_reshaped)
    predictions=[np.argmax(pred) for pred in predictions_vector]

    cnt=0
    for i in range(len(y_test)):
        if y_test[i] != predictions[i]:
            cnt+=1
            #print(f"No:{i}\nThe label is {y_test[i]}\nThe prediction is {predictions[i]}")
    print(f"The accuracy is {(len(predictions)-cnt)/len(predictions)}")

if __name__ == '__main__':
    main()

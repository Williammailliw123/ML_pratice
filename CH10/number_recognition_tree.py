import numpy as np
from tensorflow import keras
from sklearn.tree import DecisionTreeClassifier

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train_reshaped = x_train.reshape(-1,28*28)
    x_test_reshaped = x_test.reshape(-1,28*28)
    decision_tree = DecisionTreeClassifier(max_depth=1000, min_samples_leaf=5,min_samples_split=5,criterion='entropy',)
    decision_tree.fit(x_train_reshaped,y_train)
    predictions=decision_tree.predict(x_test_reshaped)
    cnt=0
    for i in range(len(y_test)):
        if y_test[i] != predictions[i]:
            cnt+=1
    print(f"The accuracy is {(len(predictions)-cnt)/len(predictions)}")

if __name__ == '__main__':
    main()

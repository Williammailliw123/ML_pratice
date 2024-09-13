
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.svm import SVC

def linear():
    linear_data = pd.read_csv('./CH11/linear.csv')
    features = np.array(linear_data[['x_1', 'x_2']])
    labels = np.array(linear_data['y'])
    utils.plot_points(features, labels)
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(features, labels)
    print("Accuracy:", svm_linear.score(features, labels))
    utils.plot_model(features, labels, svm_linear)
    # C = 0.01
    svm_c_001 = SVC(kernel='linear', C=0.01)
    svm_c_001.fit(features, labels)
    print("C = 0.01")
    print("Accuracy:", svm_c_001.score(features, labels))
    utils.plot_model(features, labels, svm_c_001)

    # C = 100
    svm_c_100 = SVC(kernel='linear', C=100)
    svm_c_100.fit(features, labels)
    print("C = 100")
    print("Accuracy:", svm_c_100.score(features, labels))
    utils.plot_model(features, labels, svm_c_100)

def one_circle():
    circular_data = pd.read_csv('./CH11/one_circle.csv')
    features = np.array(circular_data[['x_1', 'x_2']])
    labels = np.array(circular_data['y'])
    utils.plot_points(features, labels)
    # Degree = 2
    svm_degree_2 = SVC(kernel='poly', degree=2)
    svm_degree_2.fit(features, labels)
    print("Polynomial kernel of degree = 2")
    print("Accuracy:", svm_degree_2.score(features, labels))
    utils.plot_model(features, labels, svm_degree_2)

    # Degree = 4
    svm_degree_4 = SVC(kernel='poly', degree=4)
    svm_degree_4.fit(features, labels)
    print("Polynomial kernel of degree = 4")
    print("Accuracy:", svm_degree_4.score(features, labels))
    utils.plot_model(features, labels, svm_degree_4)

def two_circle():
    two_circles_data = pd.read_csv('./CH11/two_circles.csv')
    features = np.array(two_circles_data[['x_1', 'x_2']])
    labels = np.array(two_circles_data['y'])
    utils.plot_points(features, labels)
    # gamma = 0.1
    svm_gamma_01 = SVC(kernel='rbf', gamma=0.1)
    svm_gamma_01.fit(features, labels)
    print("Gamma = 0.1")
    print("Accuracy:", svm_gamma_01.score(features, labels))
    utils.plot_model(features, labels, svm_gamma_01)

    # gamma = 1
    svm_gamma_1 = SVC(kernel='rbf', gamma=1)
    svm_gamma_1.fit(features, labels)
    print("Gamma = 1")
    print("Accuracy:", svm_gamma_1.score(features, labels))
    utils.plot_model(features, labels, svm_gamma_1)

    # gamma = 10
    svm_gamma_10 = SVC(kernel='rbf', gamma=10)
    svm_gamma_10.fit(features, labels)
    print("Gamma = 10")
    print("Accuracy:", svm_gamma_10.score(features, labels))
    utils.plot_model(features, labels, svm_gamma_10)

    # gamma = 100
    svm_gamma_100 = SVC(kernel='rbf', gamma=100)
    svm_gamma_100.fit(features, labels)
    print("Gamma = 100")
    print("Accuracy:", svm_gamma_100.score(features, labels))
    utils.plot_model(features, labels, svm_gamma_100)

def main():
    #linear()
    #one_circle()
    two_circle()

if __name__ == '__main__':
    main()
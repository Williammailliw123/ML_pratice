import numpy as np
import matplotlib.pyplot as plt
import random

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def score(weights,features,bias):
    return np.dot(weights,features)+bias

def prediction(weights,features,bias):
    return sigmoid(score(weights,features,bias))

def log_loss(weights,features,bias,labels):
    pred=prediction(weights,features,bias)
    return -labels*np.log(pred)-(1-labels)*np.log(1-pred)

def total_log_loss(weights,features,bias,labels):
    total_error=0
    for i in range(len(features)):
        total_error+=log_loss(weights,features[i],bias,labels[i])
    return total_error

def logistic_trick(weights,features,bias,labels,learning_rate):
    pred=prediction(weights,features,bias)
    for i in range(len(weights)):
        weights[i]+=learning_rate*(labels-pred)*features[i]
    bias+=learning_rate*(labels-pred)
    return weights,bias

def logistic_regression_algorithm(features,labels,learning_rate,epochs):
    plt.title("Initial data")
    plt.scatter(features[:4,0],features[:4,1],color='r')
    plt.scatter(features[4:,0],features[4:,1],color='y')
    plt.show()
    plt.clf()
    weights=[1.0 for i in range(len(features[0]))]
    bias=0.0
    errors=[]
    for i in range(epochs):
        errors.append(total_log_loss(weights,features,bias,labels))
        j=random.randint(0,len(features)-1)
        weights,bias=logistic_trick(weights,features[j],bias,labels[j],learning_rate)
    plt.title("Error plot")
    plt.plot([i for i in range(epochs)],errors)
    plt.show()
    return weights,bias

def main():
    features=np.array([[1,0],[0,2],[1,1],[1,2],[1,3],[2,2],[2,3],[3,2]])
    labels=np.array([0,0,0,0,1,1,1,1])
    weights,bias=logistic_regression_algorithm(features,labels,0.01,1000)
    print(f"weights:{weights} bias:{bias}")
    
if __name__ == '__main__':
    main()
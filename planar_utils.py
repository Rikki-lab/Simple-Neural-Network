import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from testCases import *

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def layer_sizes(X,Y):
    n_x = X.shape[0]  # 输入层
    n_h = 4  # ，隐藏层，硬编码为4
    n_y = Y.shape[0]  # 输出层
    return (n_x, n_h, n_y)

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower
    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

def buildwb(n_x , n_h ,n_y):
    W1=np.random.randn(n_h,n_x) *0.01   #n_x - 输入层节点的数量
    W2=np.random.randn(n_y,n_h) *0.01   # n_h - 隐藏层节点的数量
    b1=np.zeros(shape=(n_h,1))           #n_y - 输出层节点的数量
    b2=np.zeros(shape=(n_y,1))
    parameters={"W1":W1,"W2":W2,"b1":b1,"b2":b2}
    return parameters

def forward(X,parameters):
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    Z1=np.dot(W1,X)+b1
    A1=sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache={"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}
    return A2, cache

def costfuction(A2,Y,parameters):
    m=Y.shape[1]
    singlecost=np.multiply(Y ,np.log(A2)) + np.multiply((1 - Y) , (np.log(1 - A2)))
    cost=(-1/m)*np.sum(singlecost)
    return cost

def backward(parameters,X,Y,cache):
    m=Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2= A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))#######
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,"db1": db1, "dW2": dW2, "db2": db2 }
    return grads

def renew(parameters,grads,learning_rate=1.2):
    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]
    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    return parameters

def model(X,Y,n_h,num_iterations):
    np.random.seed(3)  # 指定随机种子
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters=buildwb(n_x,n_h,n_y)

    for i in range(num_iterations):
        A2, cache = forward(X, parameters)
        cost = costfuction(A2, Y, parameters)
        grads = backward(parameters, X, Y,cache)
        parameters = renew(parameters, grads, learning_rate=0.5)
        if i % 100 == 0:
            print("第", i, "次循环, loss：" + str(cost))
    return parameters

def predict(parameters, X):
    A2, cache = forward(X, parameters)
    predictions = np.round(A2)
    return predictions
    A2, cache = forward(X, parameters)
    predictions = np.round(A2)
    return predictions




if __name__ == '__main__':
    X,Y=load_planar_dataset()#X是坐标，Y是标签
    print(np.shape(X))
    print(np.shape(Y))

    parameters = model(X, Y, n_h=4, num_iterations=1000)
    predictions = predict(parameters, X)
    print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    #二分类不行，要用神经网络




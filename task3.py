import pickle
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm


data = pickle.load(open("ytc_py.pkl", 'rb'))
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
N = 10

print("Number of training image-sets: ", len(X_train))
print("Number of testing image-sets: ", len(X_test))
print("Feature dimension of each image: ", X_train[0].shape[0])

matrix = np.empty((0, 400), float)

for i in tqdm(range(0, len(X_train), 3)):
    X_0 = X_train[i] / 255.0
    X_1 = X_train[i + 1] / 255.0
    X_2 = X_train[i + 2] / 255.0
    X = np.concatenate([X_0.T, X_1.T, X_2.T])
    p = X[0]
    for j in range(1,len(X.T[0])):
        p = p + X[j]
    p = p / len(X.T[0])
    X = (X-p)
    X = X.T
    Y = np.empty((400, 400), float)
    for j in range(len(X.T[0])):
        for k in range(len(X.T[0])):
            s = X[j]
            t = X[k]
            Y[j,k] = (1+np.dot(s, t)) ** 3.1
            if (np.isnan(Y[j,k])):
                Y[j,k] = 0
    W, v = LA.eig(Y)
    a = v.T[0:N]
    matrix = np.vstack((matrix, a))

print(matrix.shape) # 470 * 400 = 47*10 * 400

correct = 0

for i in tqdm(range(len(X_test))):
    x = X_test[i] / 255.0
    p = x[0]
    for j in range(1,len(x.T[0])):
        p = p + x[j]
    p = p / len(X.T[0])
    x = x - p
    y = np.empty((400, 400), float)
    for j in range(len(x.T[0])):
        for k in range(len(x.T[0])):
            y[j,k] = (1+np.dot(x[j], x[k])) ** 3.1
            if (np.isnan(y[j,k])):
                y[j,k] = 0

    W, v = LA.eig(y)
    s = v.T[0:N]
    ans = 0
    ans_idx = 0
    for j in range(0, len(matrix.T[0]), N):
        a = matrix[j:j+N]
        ret = np.dot(np.dot(np.dot(a, s.T), s), a.T)
        W, v = LA.eig(ret)
        mx = np.max(W)
        if (ans < mx):
            ans = mx
            ans_idx = j
    
    if (y_train[3*int(ans_idx/N)] == y_test[i]):
        correct = correct + 1

print("ans = " + str(correct / len(X_test)))
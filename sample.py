import pickle
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA


data = pickle.load(open("ytc_py.pkl", 'rb'))
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
N = 10

print("Number of training image-sets: ", len(X_train))
print("Number of testing image-sets: ", len(X_test))
print("Feature dimension of each image: ", X_train[0].shape[0])
pca = PCA(n_components=N)
matrix = np.empty((0, 400), float)

for i in range(len(X_train)):
    X = X_train[i].T
    X = X / 255.0
    pca.fit(X)
    cv = pca.get_covariance()
    W, v = LA.eig(cv)
    matrix = np.vstack((matrix, v.T[0:N]))

print(matrix.shape) # 1410 * 400 = 3 * 47*10 * 400

correct = 0

for i in range(len(X_test)):
    x = X_test[i]
    x = x / 255.0
    pca.fit(x.T)
    cv = pca.get_covariance()
    W, v = LA.eig(cv)
    s = v.T[0:N]
    ans = 0
    ans_idx = 0

    for j in range(N):
        for k in range(len(matrix.T[0])):
            ret = np.dot(s[j], matrix[k])
            if (ans < ret):
                ans = ret
                ans_idx = k
    
    if (y_train[int(ans_idx/10)] == y_test[i]):
        correct = correct + 1

print("ans = " + str(correct / len(X_test)))
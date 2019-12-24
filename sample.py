import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA


data = pickle.load(open("ytc_py.pkl", 'rb'))
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
N = 20

print("Number of training image-sets: ", len(X_train))
print("Number of testing image-sets: ", len(X_test))
print("Feature dimension of each image: ", X_train[0].shape[0])
pca = PCA(n_components=N)
matrix = np.empty((0, 400), float)
for i in range(0, len(X_train), 3):
    # ある特定人物の時系列データセット
    # len(x_0[0])  長さが取れる．
    X_0 = X_train[i]
    X_1 = X_train[i + 1]
    X_2 = X_train[i + 2]
    X = np.concatenate([X_0.T, X_1.T, X_2.T])
    X = X / 255.0
    matrix = np.vstack((matrix, pca.fit(X).components_))

print(matrix.shape)

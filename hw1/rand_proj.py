import numpy as np
def rand_proj(X, d):
    G = np.random.normal(0, 1, d*len(X[0])).reshape(len(X[0]), d)
    new_X = np.matmul(X, G)
    return new_X

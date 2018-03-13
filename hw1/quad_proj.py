import numpy as np

def quad_proj(X):
    new_X = np.column_stack((X, map(lambda X_row: map(lambda x: x * x, X_row), X)))
    for i in range(0, 63):
        new_X = np.column_stack((new_X, np.multiply(np.array(X[:, i+1:]), np.array(X[:, i]).reshape((1797, 1)))))
    return new_X
import numpy as np
import util

class MyLogisticReg2:
    def __init__(self):
        self.factor = .5
        self.W_size = -1
        self.train_size = -1
        self.W = []
        self.count = 0
        self.itr = 0

    def initAll(self, rows, cols):
        self.factor = .5
        self.count = 0
        self.itr = 0
        self.W_size = cols + 1
        self.W = np.random.uniform(low=-0.01, high=0.01, size=self.W_size)
        self.train_size = rows

    def reset(self):
        self.count = 0
        self.itr = self.itr + 1
        self.factor = self.factor / 10.

    def sigmod(self, x):
        y = 1 / (1 + np.exp(-x)) if x>=0 else np.exp(x) / (1 + np.exp(x))
        return y

    def get_response_class(self, y):
        return np.array(map(lambda y_ele: 1 if y_ele >= 0.5 else 0, y))

    def predict(self, x):
        X_test_nor = util.normalize_data(x)
        X_modified_test = np.vstack((np.ones(X_test_nor.shape[0]), X_test_nor.transpose()))
        y_pred = []
        for t in range(0, X_test_nor.shape[0]):
            o = 0
            for j in range(0, self.W_size):
                o = o + self.W[j] * X_modified_test[j][t]
            y = self.sigmod(o)
            y_pred.append(y)
        y_calc = self.get_response_class(y_pred)
        return y_calc

    def fit(self, X_train, y_train):
        X_train_nor = util.normalize_data(X_train)
        self.initAll(X_train_nor.shape[0], X_train_nor.shape[1])
        X_modified_train = np.vstack((np.ones(self.train_size), X_train_nor.transpose()))

        while (self.itr < 3):
            self.count = self.count + 1
            if (self.count > 1000):
                self.reset()
                self.predict(X_modified_train)
            delta_W = np.zeros(self.W_size)
            for t in range(0, self.train_size):
                o = 0
                for j in range(0, self.W_size):
                    o = o + self.W[j] * X_modified_train[j][t]
                y = self.sigmod(o)
                for j in range(0, self.W_size):
                    delta_W[j] = delta_W[j] + (y_train[t] - y) * X_modified_train[j][t]
            for j in range(0, self.W_size):
                self.W[j] = self.W[j] + self.factor * delta_W[j]
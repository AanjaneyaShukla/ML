import util
import numpy as np

class LogisticRegGen:

    def __init__(self):
        self.factor = .05
        self.W_size = -1
        self.train_size = -1
        self.W = []
        self.count = 0
        self.itr = 0
        self.class_count = -1

    def initAll(self, rows, cols, class_count):
        self.factor = .05
        self.count = 0
        self.itr = 0
        self.W_size = cols + 1
        self.class_count = class_count
        self.train_size = rows
        self.W = np.random.uniform(low=-0.01, high=0.01, size=self.W_size * self.class_count).\
                            reshape(self.class_count, self.W_size)

    def reset(self):
        self.count = 0
        self.itr = self.itr + 1
        self.factor = self.factor / 10.

    def softmax(self,o):
        e_o = np.exp(o - np.max(o))
        return e_o / e_o.sum()

    def get_response_class(self, y):
        return np.argmax(y)

    def predict(self, x):
        X_test_nor = util.normalize_data(x)
        X_modified_test = np.vstack((np.ones(X_test_nor.shape[0]), X_test_nor.transpose()))
        y_pred = []
        for t in range(0, X_test_nor.shape[0]):
            o = np.zeros(self.class_count)
            y = np.zeros(self.class_count)
            for i in range(0, self.class_count):
                for j in range(0, self.W_size):
                    o[i] = o[i] + self.W[i][j] * X_modified_test[j][t]
            y = self.softmax(o)
            y_pred.append(self.get_response_class(y))
        return y_pred

    def fit(self, X_train, y_train):
        X_train_nor = util.normalize_data(X_train)
        class_count = len(np.unique(y_train, False, False, True)[0])
        self.initAll(X_train_nor.shape[0], X_train_nor.shape[1], class_count)
        X_modified_train = np.vstack((np.ones(self.train_size), X_train_nor.transpose()))

        while (self.itr < 3):
            self.count = self.count + 1
            if (self.count > 20):
                self.reset()
            delta_W = np.zeros((self.class_count, self.W_size))
            for t in range(0, self.train_size):
                o = np.zeros(self.class_count)
                y = np.zeros(self.class_count)
                for i in range(0, self.class_count):
                    for j in range(0, self.W_size):
                        o[i] = o[i] + self.W[i][j] * X_modified_train[j][t]

                y = self.softmax(o)

                for i in range(0, self.class_count):
                    r = 0
                    if (int(y_train[t]) == i):
                        r = 1
                    for j in range(0, self.W_size):
                        delta_W[i][j] = delta_W[i][j] + (r - y[i]) * X_modified_train[j][t]

            for i in range(0, self.class_count):
                for j in range(0, self.W_size):
                    self.W[i][j] = self.W[i][j] + self.factor * delta_W[i][j]




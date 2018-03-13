import numpy as np
import util
import numpy.linalg as la

class MyFLDA2:
    def __init__(self):
        self.init()

    def init(self):
        self.mean = {}
        self.size = {}
        self.total_mean = []
        self.Sb = []
        self.Sw = []
        self.W = []
        self.out_mean = []
        self.mean1 = 0
        self.mean2 = 0
        self.z0 = 0
        self.error = .00001

    def eval_mean(self, X, y, class_val):
        for c in class_val:
            self.mean[c] = np.mean(X[y == c], axis=0)
            self.size[c] = int(np.sum(y[y == c], axis=0)/c)
        self.total_mean = np.mean(X, axis=0)

    def eval_Sb(self, X, y):
        feature_count = self.total_mean.size
        self.Sb = np.zeros((feature_count, feature_count))
        for class_val in self.mean:
            mean_mat = self.mean[class_val].reshape(feature_count, 1)
            total_mat = self.total_mean.reshape(feature_count, 1)
            self.Sb = self.Sb + self.size[class_val] * np.matmul((mean_mat - total_mat), ((mean_mat - total_mat).T))

    def eval_Sw(self, X, y):
        feature_count = self.total_mean.size
        self.Sw = np.zeros((feature_count, feature_count))
        for class_val in self.mean:
            mean_mat = self.mean[class_val].reshape(feature_count, 1)
            for X_ele in X[y == class_val]:
                X_ele = X_ele.reshape(feature_count, 1)
                self.Sw = self.Sw + np.matmul((X_ele - mean_mat), (X_ele - mean_mat).T)

    def predictTest(self, y, z0):
        y_pred = []
        for y_ele in y:
            if (y_ele > z0):
                y_pred.append(1)
            else:
                y_pred.append(-1)
        return y_pred

    def predict(self, X):
        y_pred = []
        y_pred_temp = np.matmul(self.W.T, X.T).T
        return self.predictTest(y_pred_temp, self.z0)

    def fit(self, X_train, y_train):
        self.init()
        class_val = set(y_train)
        self.eval_mean(X_train, y_train, class_val)
        #self.eval_Sb(X_train, y_train)
        self.eval_Sw(X_train, y_train)

        feature_count = self.total_mean.size
        self.W = np.matmul(la.inv(self.Sw),
                    (self.mean[1].reshape(feature_count, 1) - self.mean[-1].reshape(feature_count, 1)))
        self.mean1 = np.matmul(self.W.T, self.mean[1].reshape(feature_count, 1))
        self.mean2 = np.matmul(self.W.T, self.mean[-1].reshape(feature_count, 1))

        y_pred_temp = np.matmul(self.W.T, X_train.T).T
        error_min = 1
        for y_ele in y_pred_temp:
            #if ((y_ele > self.mean1 and y_ele < self.mean2) or (y_ele < self.mean1 and y_ele > self.mean2)):
                z0_temp = y_ele + self.error
                y_pred = self.predictTest(y_pred_temp, z0_temp)
                error_temp = util.get_error_rate(y_train, y_pred)
                if (error_min > error_temp):
                    self.z0 = z0_temp
                    error_min = error_temp
        #print error_min, self.z0, self.mean1, self.mean2
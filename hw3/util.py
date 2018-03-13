from __future__ import division
import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

error = .00001

def normalize_data(x):
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + error)

def gen_percentile_class(r, percentile):
    per = np.percentile(r, percentile)
    return map(lambda x: 1 if (x >= per) else 0, r)

def generate_Boston50():
    boston = load_boston()
    [X, r] = [boston.data, boston.target]
    y = gen_percentile_class(r, 50)
    return [X, y]


def generate_Boston75():
    #[X, r] = read_file(os.path.join(input_dir, 'housing.data'))
    boston = load_boston()
    [X, r] = [boston.data, boston.target]
    y = gen_percentile_class(r, 75)
    return [X, y]

def adding_noise(X):
    #X = X + 1
    print X

def generate_digits():
    digits = load_digits()
    return [digits.data, digits.target]

def get_error_rate(y_test, y_test_pred):
    accuracy_count = 0
    for i in range(0, len(y_test_pred)):
        if (y_test[i] == y_test_pred[i]):
            accuracy_count += 1
    return 1 - (accuracy_count / len(y_test_pred))

def linear_svc_error_rate(X_train, y_train, X_test, y_test):
    lSVC = LinearSVC()
    lSVC.fit(X_train, y_train)
    y_test_pred = lSVC.predict(X_test)
    return get_error_rate(y_test, y_test_pred)


def svc_error_rate(X_train, y_train, X_test, y_test):
    svc = SVC()
    svc.fit(X_train, y_train)
    y_test_pred = svc.predict(X_test)
    return get_error_rate(y_test, y_test_pred)


def logistic_regression_error_rate(X_train, y_train, X_test, y_test):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_test_pred = lr.predict(X_test)
    return get_error_rate(y_test, y_test_pred)

def get_train_test(input_data):
    return [input_data[:, :-1], input_data[:, -1]]




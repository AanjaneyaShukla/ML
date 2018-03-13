import numpy as np
import util
from MyLogisticReg2 import MyLogisticReg2
from LogisticRegGen import LogisticRegGen

def my_cross_val(method, X, y, k):
    input_data = np.column_stack((X, y))
    np.random.shuffle(input_data)
    k_split_input_data = np.array_split(input_data, k)
    error_rate = []

    for x in range(0, k):
        input_data_train = None
        input_data_test = k_split_input_data[x]
        for y in range(0, k):
            if x != y:
                if input_data_train is None:
                    input_data_train = k_split_input_data[y]
                else:
                    input_data_train = np.row_stack((input_data_train, k_split_input_data[y]))

        [X_train, y_train] = util.get_train_test(input_data_train)
        [X_test, y_test] = util.get_train_test(input_data_test)

        if method == 'MyLogisticReg2':
            my_logistic_reg2 = MyLogisticReg2()
            my_logistic_reg2.fit(X_train, y_train)
            y_test_pred = my_logistic_reg2.predict(X_test)
            error_rate.append(util.get_error_rate(y_test, y_test_pred))

        if method == 'LogisticRegGen':
            logisticRegGen = LogisticRegGen()
            logisticRegGen.fit(X_train, y_train)
            y_test_pred = logisticRegGen.predict(X_test)
            error_rate.append(util.get_error_rate(y_test, y_test_pred))

        if method == 'LogisticRegression':
            error_rate.append(util.logistic_regression_error_rate(X_train, y_train, X_test, y_test))

    error_mean = np.mean(error_rate)
    error_std = np.std(error_rate)

    print error_rate, error_mean, error_std
    return error_rate
import numpy as np
import random
import util

def my_train_test(method, X, y, train_fraction, k):
    train_count = int(train_fraction * len(y))
    error_rate = []
    for x in range(0, k):
        input_data = np.column_stack((X, y))
        random.shuffle(input_data)
        input_data_train = input_data[:train_count]
        input_data_test = input_data[train_count:]

        [X_train, y_train] = util.get_train_test(input_data_train)
        [X_test, y_test] = util.get_train_test(input_data_test)

        util.run_models(method, error_rate, X_train, y_train, X_test, y_test)
    error_mean = np.mean(error_rate)
    error_std = np.std(error_rate)

    print error_rate, error_mean, error_std
    return error_rate
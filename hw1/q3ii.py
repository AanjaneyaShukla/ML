import util
import my_train_test

if __name__ == "__main__":

    print '1.LinearSVC with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    my_train_test.my_train_test('LinearSVC', boston50_X, boston50_y, 0.75, 10)

    print '2.LinearSVC with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    my_train_test.my_train_test('LinearSVC', boston75_X, boston75_y, 0.75, 10)

    print '3.LinearSVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    my_train_test.my_train_test('LinearSVC', digit_X, digit_y, 0.75, 10)

    print '4.SVC with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    my_train_test.my_train_test('SVC', boston50_X, boston50_y, 0.75, 10)

    print '5.SVC with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    my_train_test.my_train_test('SVC', boston75_X, boston75_y, 0.75, 10)

    print '6.SVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    my_train_test.my_train_test('SVC', digit_X, digit_y, 0.75, 10)

    print '7.LogisticRegression with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    my_train_test.my_train_test('LogisticRegression', boston50_X, boston50_y, 0.75, 10)

    print '8.LogisticRegression with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    my_train_test.my_train_test('LogisticRegression', boston75_X, boston75_y, 0.75, 10)

    print '9.LogisticRegression with Digits'
    [digit_X, digit_y] = util.generate_digits()
    my_train_test.my_train_test('LogisticRegression', digit_X, digit_y, 0.75, 10)
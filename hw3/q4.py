import util
import my_cross_val

if __name__ == "__main__":

    print '1.LogisticRegGen with Digits'
    [digit_X, digit_y] = util.generate_digits()
    my_cross_val.my_cross_val('LogisticRegGen', digit_X, digit_y, 5)

    print '2.LogisticRegression with Digits'
    [digit_X, digit_y] = util.generate_digits()
    my_cross_val.my_cross_val('LogisticRegression', digit_X, digit_X, 5)
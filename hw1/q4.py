import util
import my_cross_val
import rand_proj
import quad_proj

if __name__ == "__main__":

    print '1.LinearSVC with X1'
    [digit_X, digit_y] = util.generate_digits()
    X1 = rand_proj.rand_proj(digit_X, 32)
    my_cross_val.my_cross_val('LinearSVC', X1, digit_y, 10)

    print '2.LinearSVC with X2'
    [digit_X, digit_y] = util.generate_digits()
    X2 = quad_proj.quad_proj(digit_X)
    my_cross_val.my_cross_val('LinearSVC', X2, digit_y, 10)

    print '3.LinearSVC with X3'
    [digit_X, digit_y] = util.generate_digits()
    X2 = quad_proj.quad_proj(digit_X)
    X3 = rand_proj.rand_proj(X2, 64)
    my_cross_val.my_cross_val('LinearSVC', X3, digit_y, 10)

    print '4.SVC with X1'
    [digit_X, digit_y] = util.generate_digits()
    X1 = rand_proj.rand_proj(digit_X, 32)
    my_cross_val.my_cross_val('SVC', X1, digit_y, 10)

    print '5.SVC with X2'
    [digit_X, digit_y] = util.generate_digits()
    X2 = quad_proj.quad_proj(digit_X)
    my_cross_val.my_cross_val('SVC', X2, digit_y, 10)

    print '6.SVC with X3'
    [digit_X, digit_y] = util.generate_digits()
    X2 = quad_proj.quad_proj(digit_X)
    X3 = rand_proj.rand_proj(X2, 64)
    my_cross_val.my_cross_val('SVC', X3, digit_y, 10)

    print '7.LogisticRegression with X1'
    [digit_X, digit_y] = util.generate_digits()
    X1 = rand_proj.rand_proj(digit_X, 32)
    my_cross_val.my_cross_val('LogisticRegression', X1, digit_y, 10)

    print '8.LogisticRegression with X2'
    [digit_X, digit_y] = util.generate_digits()
    X2 = quad_proj.quad_proj(digit_X)
    my_cross_val.my_cross_val('LogisticRegression', X2, digit_y, 10)

    print '9.LogisticRegression with X3'
    [digit_X, digit_y] = util.generate_digits()
    X2 = quad_proj.quad_proj(digit_X)
    X3 = rand_proj.rand_proj(X2, 64)
    my_cross_val.my_cross_val('LogisticRegression', X3, digit_y, 10)
import util
import my_cross_val

if __name__ == "__main__":

    print '1.MyLogisticReg2 with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    my_cross_val.my_cross_val('MyLogisticReg2', boston50_X, boston50_y, 5)

    print '2.MyLogisticReg2 with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    my_cross_val.my_cross_val('MyLogisticReg2', boston75_X, boston75_y, 5)

    print '3.LogisticRegression with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    my_cross_val.my_cross_val('LogisticRegression', boston50_X, boston50_y, 5)

    print '4.LogisticRegression with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    my_cross_val.my_cross_val('LogisticRegression', boston75_X, boston75_y, 5)

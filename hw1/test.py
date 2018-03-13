from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import rand_proj
import quad_proj
import util

def linear_svc(X, y):
    linear_svc = LinearSVC()
    cross_val = KFold(n_splits=10, shuffle=True)
    error = (1 - cross_val_score(linear_svc, X, y, scoring="accuracy", cv=cross_val))
    print (error, np.mean(error), np.std(error))

def svc(X, y):
    svc = SVC()
    cross_val = KFold(n_splits=10, shuffle=True)
    error = (1 - cross_val_score(svc, X, y, scoring="accuracy", cv=cross_val))
    print (error, np.mean(error), np.std(error))

def logisticRegression(X, y):
    logisticRegression = LogisticRegression()
    cross_val = KFold(n_splits=10, shuffle=True)
    error = (1 - cross_val_score(logisticRegression, X, y, scoring="accuracy", cv=cross_val))
    print (error, np.mean(error), np.std(error))

def q3i():
    print '1.LinearSVC with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    linear_svc(boston50_X, boston50_y)

    print '2.LinearSVC with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    linear_svc(boston75_X, boston75_y)

    print '3.LinearSVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    linear_svc(digit_X, digit_y)

    print '4.SVC with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    svc(boston50_X, boston50_y)

    print '5.SVC with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    svc(boston75_X, boston75_y)

    print '6.SVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    svc(digit_X, digit_y)

    print '7.LogisticRegression with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    logisticRegression(boston50_X, boston50_y)

    print '8.LogisticRegression with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    logisticRegression(boston75_X, boston75_y)

    print '9.LogisticRegression with Digits'
    [digit_X, digit_y] = util.generate_digits()
    logisticRegression(digit_X, digit_y)

def q3ii():
    print '1.LinearSVC with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    linear_svc(boston50_X, boston50_y)

    print '2.LinearSVC with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    linear_svc(boston75_X, boston75_y)

    print '3.LinearSVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    linear_svc(digit_X, digit_y)

    print '4.SVC with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    svc(boston50_X, boston50_y)

    print '5.SVC with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    svc(boston75_X, boston75_y)

    print '6.SVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    svc(digit_X, digit_y)

    print '7.LogisticRegression with Boston50'
    [boston50_X, boston50_y] = util.generate_Boston50()
    logisticRegression(boston50_X, boston50_y)

    print '8.LogisticRegression with Boston75'
    [boston75_X, boston75_y] = util.generate_Boston75()
    logisticRegression(boston75_X, boston75_y)

    print '9.LogisticRegression with Digits'
    [digit_X, digit_y] = util.generate_digits()
    logisticRegression(digit_X, digit_y)

def q4i():
    print '1.LinearSVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    new_X = rand_proj.rand_proj(digit_X, 32)
    linear_svc(new_X, digit_y)

    print '2.SVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    new_X = rand_proj.rand_proj(digit_X, 32)
    svc(new_X, digit_y)

    print '3.LogisticRegression with Digits'
    [digit_X, digit_y] = util.generate_digits()
    new_X = rand_proj.rand_proj(digit_X, 32)
    logisticRegression(new_X, digit_y)

def q4ii():
    print '1.LinearSVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    new_X = quad_proj.quad_proj(digit_X)
    linear_svc(new_X, digit_y)

    print '2.SVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    new_X = quad_proj.quad_proj(digit_X)
    svc(new_X, digit_y)

    print '3.LogisticRegression with Digits'
    [digit_X, digit_y] = util.generate_digits()
    new_X = quad_proj.quad_proj(digit_X)
    logisticRegression(new_X, digit_y)

def q4iii():
    print '1.LinearSVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    X2 = quad_proj.quad_proj(digit_X)
    X3 = rand_proj.rand_proj(X2, 64)
    linear_svc(X3, digit_y)

    print '2.SVC with Digits'
    [digit_X, digit_y] = util.generate_digits()
    X2 = quad_proj.quad_proj(digit_X)
    X3 = rand_proj.rand_proj(X2, 64)
    svc(X3, digit_y)

    print '3.LogisticRegression with Digits'
    [digit_X, digit_y] = util.generate_digits()
    X2 = quad_proj.quad_proj(digit_X)
    X3 = rand_proj.rand_proj(X2, 64)
    logisticRegression(X3, digit_y)

if __name__ == "__main__":

    # q3i
    print 'Testing Question 3 part i'
    q3i()

    #q3ii
    print 'Testing Question 3 part ii'
    q3ii()

    #q4i
    print 'Testing Question 4 part i'
    q4i()

    #q4ii
    print 'Testing Question 4 part ii'
    q4ii()

    # q4iii
    print 'Testing Question 4 part iii'
    q4iii()
"""This module provides functions for trAdaboost.
"""


import numpy
import math
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics


def gridSearchCV(X, y, param_grid, sample_weight=None):
    """Search the best gamma and C for the model"""

    best_auc = 0
    best_clf = svm.SVC()
    cv = KFold(n_splits=10)

    for gamma in param_grid['gamma']:
        for C in param_grid['C']:
            clf = svm.SVC(gamma=gamma, C=C, probability=True)
            predict = numpy.zeros(y.shape)
            for train_indc, test_indc in cv.split(X, y):
                if sample_weight is None:
                    clf.fit(X[train_indc], y[train_indc])
                else:
                    clf.fit(X[train_indc], y[train_indc],
                            sample_weight=sample_weight[train_indc])
                predict[test_indc] = clf.predict_proba(X[test_indc])

            auc = metrics.roc_auc_score(y, predict)
            # print predict
            print "searching best auc: %f; gamma: %f, C: %f" % (auc, gamma, C)
            if auc > best_auc:
                best_auc = auc
                best_clf = clf

    print("The best auc is %f, best gamma: %f, best C: %f"
          % (best_auc, best_clf.get_params()['gamma'],
             best_clf.get_params()['C']))
    return best_clf


def trAdaboost(Td, Ts, labeld, labels, S, N):
    """This is a function provides trAdaboost algorithm.
    Paramaters:

    Td: numpy
        source training data
    Ts: numpy
        target training data
    labeld: vector
        source label
    labels: vector
        target label
    S: numpy
        testing data
    N: int
        iterater times
    """

    # get the length of all labeled data

    # t = Td.shape[0] + Ts.shape[0]
    n = Td.shape[0]

    # init weight vector
    wd = numpy.ones([Td.shape[0], ])
    ws = numpy.ones([Ts.shape[0], ])
    ht = numpy.zeros([N, S.shape[0]])

    beta = 0
    beta_t = numpy.zeros([1, N])

    # init some paramaters
    C_range = numpy.logspace(-2, 10, 13)
    gamma_range = numpy.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)

    # StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for i in range(N):
        sample_weight = (numpy.hstack((wd, ws))
                         / float(numpy.sum(ws) + numpy.sum(wd)))
        best_clf_model = gridSearchCV(numpy.vstack((Td, Ts)),
                                      numpy.hstack((labeld, labels)),
                                      param_grid=param_grid,
                                      sample_weight=sample_weight)

        hs = best_clf_model.predict(Ts)
        hd = best_clf_model.predict(Td)
        ht[i, :] = best_clf_model.predict(S)

        # TODO: error_t suppost to be less than 1/2
        error_t = numpy.dot(numpy.abs(hs - labels), ws) / numpy.sum(ws)
        print "The overall error_t is %f" % error_t

        beta_t[0, i] = error_t / (1 - error_t)
        beta = 1. / (1 + math.sqrt(2. * math.log(float(n)/N)))

        wd = wd * (beta ** abs(hd - labeld))
        ws = ws * (beta_t[0, i] ** -abs(hs - labels))

    hf = numpy.zeros([1, S.shape[0]])
    base = numpy.prod(beta_t[0, int(math.ceil(N / 2)):] ** -0.5)
    for i in range(S.shape[0]):
        # production of this powers
        posibity = numpy.prod(beta_t[0, int(math.ceil(N / 2)):] **
                              -ht[int(math.ceil(N / 2)):, i])

        if(posibity > base):
            hf[0, i] = 1
        else:
            hf[0, i] = 0

    return hf

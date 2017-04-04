"""This module provides functions for trAdaboost.
"""


import numpy
import math
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics
from multiprocessing import Pool, Array
import os
import sys

def svm_thread(X, y, split, gamma, C, sample_weight=None):

    clf = svm.SVC(gamma=gamma, C=C, probability=True)
    predict = numpy.zeros(y.shape)
    for train_indc, test_indc in split:
        if sample_weight is None:
            clf.fit(X[train_indc], y[train_indc])
        else:
            clf.fit(X[train_indc], y[train_indc],
                    sample_weight=sample_weight[train_indc])
        predict[test_indc] = clf.predict_proba(X[test_indc])[:, 0]

    auc = metrics.roc_auc_score(y, predict)
    # print predict
    if os.environ['debug']:
        print ("searching best auc: %f; gamma: %f, C: %f"
               % (auc, gamma, C))
    
    return [auc, gamma, C, clf]

def cross_validation(X, y):
    """standard cross_validation method which uses AUC as score"""
    
    C_range = 2. ** numpy.arange(-5, 15)
    gamma_range = 2. ** numpy.arange(-20, -5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    
    kf = KFold(n_splits=10)
    predict = numpy.zeros(y.shape)
    for train, test in kf.split(X, y):
        best_clf = gridSearch(X[train], y[train], param_grid)
        #best_clf.fit(X[train], y[train])
        predict[test] = best_clf.predict_proba(X[test])[:, 0]

    return predict

def gridSearch(X, y, param_grid, sample_weight=None):
    """Grid search for svm params includes gamma and C. 
    No cross validation applied. 
    
    This method will improve the effency of the whole learning process"""
    
    pool = Pool()
    results = []
    for gamma in param_grid['gamma']:
        for C in param_grid['C']:
            results.append(pool.apply_async(svm_thread,
                                            args=(X, y,
                                                  [(range(y.shape[0]),
                                                   range(y.shape[0]))],
                                                  gamma, C, sample_weight)))

    bests = [0, 0, 0]
    #progressBar = progressbar.ProgressBar(maxval=len(results)).start()
    printProgressBar(0, len(results))
    i = 0
    for result in results:
        part = result.get()
        if part[0] > bests[0]:
            bests = part
        i += 1
        #progressBar.update(i)
        printProgressBar(i, len(results))

    pool.close()
    pool.join()

    print("The best auc is %f, best gamma: %f, best C: %f"
          % (bests[0], bests[1],bests[2]))

    #progressBar.finish()
    return bests[3]


def gridSearchCV(X, y, param_grid, sample_weight=None):
    """Search the best gamma and C for the model"""

    bests = [0, 0, 0]
    cv = KFold(n_splits=10)

    split = list(cv.split(X, y))

    pool = Pool()

    results = []
    for gamma in param_grid['gamma']:
        for C in param_grid['C']:
            results.append(pool.apply_async(svm_thread,
                                            args=(X, y, split, gamma, C,
                                                  sample_weight,)))

    #progressBar = progressbar.ProgressBar(maxval=len(results)).start()
    printProgressBar(0, len(results))
    i = 0
    for result in results:
        part = result.get()
        if part[0] > bests[0]:
            bests = part
        i += 1
        printProgressBar(i, len(results))

    pool.close()
    pool.join()

    print("The best auc is %f, best gamma: %f, best C: %f"
          % (bests[0], bests[1],bests[2]))

    return bests[3]


def trAdaboost(Td, Ts, labeld, labels, S, N, preset_model=None):
    """This is a function provides trAdaboost algorithm.
    Paramaters:

    Td: numpy
        Source training data
    Ts: numpy
        Target training data
    labeld: vector
        Source label
    labels: vector
        Target label
    S: numpy
        Testing data
    N: int
        Iterater times
    preset_model: SVC classifier
        Set a preset model for all iteration. If not set, trAdaboost will try to
        find the best model for each iteration.
    """

    # get the length of all labeled data

    # t = Td.shape[0] + Ts.shape[0]
    n = Td.shape[0]

    # init weight vector
    # balance positive and negative instances
    diff_positive_num = sum(labeld)
    diff_negative_num = labeld.shape[0] - diff_positive_num
    same_positive_num = sum(labels)
    same_negative_num = labels.shape[0] - same_positive_num
    wd = numpy.zeros([Td.shape[0], ])
    wd[labeld == 1] = 1.0 / diff_positive_num
    wd[labeld == 0] = 1.0 / diff_negative_num
    ws = numpy.ones([Ts.shape[0], ])
    ws[labels == 1] = 1.0 / same_positive_num
    ws[labels == 0] = 1.0 / same_negative_num
    ht = numpy.zeros([N, S.shape[0]])

    beta = 0
    beta_t = numpy.zeros([1, N])

    # init some paramaters
    C_range = 2. ** numpy.arange(-5, 15)
    gamma_range = 2. ** numpy.arange(-20, -5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    
    # StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for i in range(N):
        print "iterator %d" % i
    
        sample_weight = (numpy.hstack((wd, ws))
                         / float(numpy.sum(ws) + numpy.sum(wd)))

        # find a best model for each iterator or stick to a preset model
        if preset_model is None:
            best_clf_model = gridSearch(numpy.vstack((Td, Ts)),
                                          numpy.hstack((labeld, labels)),
                                          param_grid=param_grid,
                                          sample_weight=sample_weight)
        else:
            best_clf_model = preset_model

        #best_clf_model.fit(numpy.vstack((Td, Ts)),
        #                   numpy.hstack((labeld, labels)),
        #                   sample_weight=sample_weight)
        hs = best_clf_model.predict_proba(Ts)[:, 0]
        hd = best_clf_model.predict_proba(Td)[:, 0]
        ht[i, :] = best_clf_model.predict_proba(S)[:, 0]

        # TODO: error_t suppost to be less than 1/2
        error_t = numpy.dot(numpy.abs(hs - labels), ws) / numpy.sum(ws)
        print "The iterator error_t is %f" % error_t

        beta_t[0, i] = error_t / (1 - error_t)
        beta = 1. / (1 + math.sqrt(2. * math.log(float(n)/N)))

        wd = wd * (beta ** abs(hd - labeld))
        ws = ws * (beta_t[0, i] ** -abs(hs - labels))

    hf = numpy.zeros([1, S.shape[0]])
    base = numpy.prod(beta_t[0, int(math.ceil(N / 2)):] ** -0.5)
    posibity = numpy.zeros([S.shape[0], ])
    for i in range(S.shape[0]):
        # production of this powers
        posibity[i] = numpy.prod(beta_t[0, int(math.ceil(N / 2)):] **
                                 -ht[int(math.ceil(N / 2)):, i].T)
        #print posibity[i]

        if(posibity[i] > base):
            hf[0, i] = 1
        else:
            hf[0, i] = 0

    return posibity / base

def printProgressBar(iteration, total,
                     prefix = '', suffix = '',
                     decimals = 1, length = 50, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : 
                      positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration) /
                                                     float(total))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s\r' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complet
    # if iteration == total:
    #     print()

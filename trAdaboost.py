"""This module provides functions for trAdaboost.
"""

def trAdaboost(Td, Ts, labeld, labels, S, N):
    """This is a function provides trAdaboost algorithm.
    """
    import numpy
    import math
    from sklearn import svm

    # get the length of all labeled data

    t = Td.shape[0] + Ts.shape[0]
    n = Td.shape[0]

    # init weight vector
    wd = numpy.ones([Td.shape[0], 1])
    ws = numpy.ones([Ts.shape[0], 1])
    ht = numpy.zeros([N, S.shape[0]])

    beta = 0
    beta_t = numpy.zeros([1, N])
    for i in range(N):
        clf_weights = svm.SVC()
        clf_weights.fit(numpy.vstack((Td, Ts)), numpy.hstask((labeld, labels)),
                        sample_weight=numpy.hstask((wd, ws)) / (numpy.sum(ws) + numpy.sum(wd)))
        hs = clf_weights.predict(Ts)
        hd = clf_weights.predict(Td)
        ht[i, :] = clf_weights.predict(S)
        # TODO: error_t suppost to be less than 1/2
        error_t = numpy.dot(numpy.abs(hs - labels), ws) / numpy.sum(ws)

        beta_t[1, i] = error_t / (1 - error_t)
        beta = 1. / (1 + sqrt(2. * ln(float(n) / N)))

        wd = wd * (beta ** abs(hd - labeld))
        ws = ws * (beta_t[1, i] ** -abs(hs - labels))

    hf = numpy.zeros([1, S.shape[0]])
    base = numpy.prod(beta_t[1, math.ceil(N / 2) : N] ** -0.5)
    for i in range(S.shape[0]):
        # production of this powers
        posibity = numpy.prod(beta_t[1, math.ceil(N / 2) : N] ** -ht[i, math.ceil(N / 2) : N])

        if(posibity > base):
            hf[1, i] = 1
        else:
            hf[1, i] = 0

    return hf
        

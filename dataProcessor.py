"""Load data from scratch and do some pre-process
"""

import scipy.io
import numpy


def loadExpression(data):
    """Loads data from specified matlab mat file whose path is
       set to `data/express/matlab.mat`.

    Paramaters:
    ------------------------------------------------
    data: dict
        data will be filled with loaded matrices.
    """
    mat = scipy.io.loadmat('data/express/matlab.mat')
    data['express'] = mat['exp_matrix']
    data['label'] = numpy.logical_or(
        mat['S1'][:, 3] == 'Stable Disease',
        mat['S1'][:, 3] == 'Clinical Progressive Disease')
    data['uid'] = mat['S1'][:, 1]
    data['drug'] = mat['S1'][:, 2]
    data['disease'] = mat['S1'][:, 0]


def loadCNV(data):
    """Load CNV data from mat file. Path is set to `data/CNV/CNV.mat`.

    Paramaters:
    ------------------------------------------------
    data: dict
        This paramater should be initialized by :func:`loadExpression`
        first, for the patients id is loaded by :func:`loadExpression`.
    """
    mat = scipy.io.loadmat('data/CNV/CNV.mat')

    patients = mat['U'][:, 1]
    cnv = mat['M']

    cnvdata = numpy.zeros([data['uid'].shape[0], cnv.shape[0]])
    found = False
    count = 0.0
    for i in range(data['uid'].shape[0]):
        for j in range(patients.shape[0]):
            if patients[j][0][0:12] == data['uid'][i][0]:
                cnvdata[i] = cnv[:, j]
                found = True
                count += 1
                break

        if not found:
            print("warning: not found %s" % data['uid'][i][0])

        found = False

    print("found percentage %f" % (count/data['uid'].shape[0]))

    data['cnv'] = cnvdata


def test():
    data = {}
    loadExpression(data)
    loadCNV(data)

    print data


if __name__ == "__main__":
    test()

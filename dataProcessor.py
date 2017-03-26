"""Load data from scratch and do some pre-process
"""

import scipy.io
import os
import subprocess
import numpy
from trAdaboost import trAdaboost
from trAdaboost import gridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn import metrics


def loadExpression(data):
    """Loads data from specified matlab mat file whose path is
       set to `../data/express/matlab.mat`.

    Paramaters:
    ------------------------------------------------
    data: dict
        data will be filled with loaded matrices.
    """
    mat = scipy.io.loadmat('../data/express/matlab.mat')
    data['express'] = mat['exp_matrix']
    data['label'] = numpy.logical_or(
        mat['S1'][:, 3] == 'Stable Disease',
        mat['S1'][:, 3] == 'Clinical Progressive Disease')
    data['uid'] = mat['S1'][:, 1]
    data['drug'] = mat['S1'][:, 2]
    data['disease'] = mat['S1'][:, 0]


def loadCNV(data):
    """Load CNV data from mat file. Path is set to `../data/CNV/CNV.mat`.

    Paramaters:
    ------------------------------------------------
    data: dict
        This paramater should be initialized by :func:`loadExpression`
        first, for the patients id is loaded by :func:`loadExpression`.
    """
    mat = scipy.io.loadmat('../data/CNV/CNV.mat')

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


def data_filter(data, drug_name, disease_name=None):
    """Filter instances using `drug_name` from `data`.
    Returned a new dict contains all filtered data.
    """
    filter = data['drug'] == drug_name
    if disease_name != None:
        filter = numpy.logical_and(filter, data['disease'] == diseaes_name)

    filtered_data = {}
    filtered_data['uid'] = data['uid'][filter]
    filtered_data['express'] = data['express'][filter, :]
    filtered_data['label'] = data['label'][filter]
    filtered_data['disease'] = data['disease'][filter]

    return filtered_data
    

def test():
    data = {}
    loadExpression(data)

    # normalize gene express based on all data
    expression = data['express']
    expression = expression[:, expression.std(0) != 0]
    expression = (expression -
                  numpy.resize(expression.mean(0),
                               expression.shape)) / numpy.resize(expression.std(0), expression.shape)
    expression[numpy.logical_or(expression > 2, expression < -2)] = 0
    data['express'] = preprocessing.MinMaxScaler().fit_transform(expression)

    # filter two drugs as source and target data
    source_data = data_filter(data, 'Cisplatin')
    target_data = data_filter(data, 'Carboplatin')

    source_express = source_data['express']
    target_express = target_data['express']

    # do feature selction by mrmr
    if(os.path.isfile('~feature.txt')):
        print "Feature file found, use existing features"
        feature_indc = numpy.loadtxt('~feature.txt').astype('int')
    else:
        print "Feature file not found, feature selection by mrmr"
        numpy.savetxt("~temp.csv", numpy.ceil(source_express*100), fmt="%.3f", delimiter=',')
        output = subprocess.check_output(["mrmr_c_src/mrmr", "-i", "temp.csv", "-n", "200", "-s", "4000", "-v", "21000"])
        # load features from data
        table = numpy.fromstring(output).reshape(200, 4)
        feature_indc = table[:, 1].astype('int')
        numpy.savetxt('~feature.txt', feature_indc)
        print "Feature selection finished"

    source_express = source_express[:, feature_indc]
    target_express = target_express[:, feature_indc]
    print "Instance number, source: %d, target: %d" % (source_express.shape[0], target_express.shape[0])

    source_label = numpy.zeros(source_data['label'].shape)
    source_label[source_data['label']] = 1
    target_label = numpy.zeros(target_data['label'].shape)
    target_label[target_data['label']] = 1
    
    kf = KFold(n_splits = 10)
    predict = numpy.zeros(target_label.shape)

    C_range = 2. ** numpy.arange(-2, 4, 0.5)
    gamma_range = 2. ** numpy.arange(-2, 4, 0.5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    best_clf = gridSearchCV(source_express, source_label, param_grid=param_grid, cv=cv)
    return
    for train, test in kf.split(target_label):
        print "training one split"
        predict[test] = trAdaboost(source_express, target_express[train],
                             source_label, target_label[train],
                             target_express[test], 10)
        
    print "The overall trAdaboost AUC is %f" % metrics.roc_auc_score(target_label, predict)


if __name__ == "__main__":
    test()

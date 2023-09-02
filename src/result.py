import numpy as np
import tensorflow
from keras.models import load_model
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, auc)

from processing import kdd_encoding
from unsw import unsw_encoding
import numpy as np
from sklearnex import patch_sklearn
from daal4py.oneapi import sycl_context
patch_sklearn()

from sklearn.cluster import DBSCAN

params = {'train_data': 494021, 'features_nb': 4,
          'batch_size': 1024, 'encoder': 'standarscaler',
          'dataset': 'kdd'}

model_name = './models/' + '494021_4_mse_nadam_sigmoid_1_128_1024' + \
    '_0.2_CuDNNLSTM_standarscaler_1562685990.8704927st'



def load_data():
    if params['dataset'] == 'kdd':
        x_train, x_test, y_train, y_test = kdd_encoding(params)
    elif params['dataset'] == 'unsw':
        x_train, x_test, y_train, y_test = unsw_encoding(params)

 
    x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
    x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])
    return x_train, x_test, y_train, y_test


def print_results(params, model, x_train, x_test, y_train, y_test):
    print('Val loss and acc:')
    print(model.evaluate(x_test, y_test, params['batch_size']))

    y_pred = model.predict(x_test, params['batch_size'])

    print('\nConfusion Matrix:')
    conf_matrix = confusion_matrix(y_test.argmax(axis=1),
                                   y_pred.argmax(axis=1))
    print(conf_matrix)

    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix) 
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)  
    TP = np.diag(conf_matrix) 
    TN = conf_matrix.sum() - (FP + FN + FP)  

    print('\nTPR:')  
    
    print(TP / (TP + FN))

    print('\nFPR:') 
    print(FP / (FP + TN))

    cost_matrix = [[0, 1, 2, 2, 2],
                   [1, 0, 2, 2, 2],
                   [2, 1, 0, 2, 2],
                   [4, 2, 2, 0, 2],
                   [4, 2, 2, 2, 0]]

    tmp_matrix = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            tmp_matrix[i][j] = conf_matrix[i][j] * cost_matrix[i][j]

    print('\nCost:')
    print(tmp_matrix.sum()/conf_matrix.sum())

    print('\nAUC:')  
    print(roc_auc_score(y_true=y_test, y_score=y_pred, average=None))

    print('\nPrecision:')  

    print(precision_score(y_true=y_test.argmax(axis=1),
                          y_pred=y_pred.argmax(axis=1), average=None))


if __name__ == "__main__":
   

    model = load_model(model_name)
    model.summary()

    x_train, x_test, y_train, y_test = load_data()
    print_results(params, model, x_train, x_test, y_train, y_test)

import numpy as np
from fastapi import FastAPI
from keras.models import load_model
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, auc)

from processing import kdd_encoding
from unsw import unsw_encoding

app = FastAPI()

params = {'train_data': 494021, 'features_nb': 4,
          'batch_size': 1024, 'encoder': 'standardscaler',
          'dataset': 'kdd'}

model_name = './models/' + '494021_4_mse_nadam_sigmoid_1_128_1024' + \
    '_0.2_CuDNNLSTM_standardscaler_1562685990.8704927st'

def load_data():
    if params['dataset'] == 'kdd':
        x_train, x_test, y_train, y_test = kdd_encoding(params)
    elif params['dataset'] == 'unsw':
        x_train, x_test, y_train, y_test = unsw_encoding(params)

    x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
    x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])
    return x_train, x_test, y_train, y_test

def evaluate_model():
    model = load_model(model_name)
    x_train, x_test, y_train, y_test = load_data()
    val_loss, val_acc = model.evaluate(x_test, y_test, params['batch_size'])
    
    y_pred = model.predict(x_test, params['batch_size'])
    
    conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix) 
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)  
    TP = np.diag(conf_matrix) 
    TN = conf_matrix.sum() - (FP + FN + FP)  
    
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    cost_matrix = [[0, 1, 2, 2, 2],
                   [1, 0, 2, 2, 2],
                   [2, 1, 0, 2, 2],
                   [4, 2, 2, 0, 2],
                   [4, 2, 2, 2, 0]]

    tmp_matrix = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            tmp_matrix[i][j] = conf_matrix[i][j] * cost_matrix[i][j]

    cost = tmp_matrix.sum() / conf_matrix.sum()
    auc_score = roc_auc_score(y_true=y_test, y_score=y_pred, average=None)
    
    precision = precision_score(y_true=y_test.argmax(axis=1),
                                y_pred=y_pred.argmax(axis=1), average=None)
    
    return {
        "Validation Loss": val_loss,
        "Validation Accuracy": val_acc,
        "Confusion Matrix": conf_matrix.tolist(),
        "True Positive Rate (TPR)": TPR.tolist(),
        "False Positive Rate (FPR)": FPR.tolist(),
        "Cost": cost,
        "AUC Scores": auc_score.tolist(),
        "Precision Scores": precision.tolist()
    }

@app.get("/evaluate_model")
def get_evaluation_results():
    evaluation_results = evaluate_model()
    return evaluation_results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

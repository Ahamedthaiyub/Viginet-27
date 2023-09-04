import tensorflow
import pandas as pd
import numpy as np
import os
from time import time
from keras.layers import Dense, Dropout, LSTM, GRU

from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint

from processing import kdd_encoding
from unsw import unsw_encoding
from result import print_results


csv_values = ['epochs', 'acc', 'loss', 'val_acc', 'val_loss', "train_data",
              "features_nb", 'loss_fct', 'optimizer', 'activation_fct',
              'layer_nb', 'unit_nb', 'batch_size', 'dropout', 'cell_type',
              'encoder']

csv_best_res = ['param', 'value', 'min_mean_val_loss']


params = {'epochs': 3, 'train_data': 494021, 'features_nb': 4,
          'loss_fct': 'mse', 'optimizer': 'rmsprop',
          'activation_fct': 'sigmoid', 'layer_nb': 1, 'unit_nb': 128,
          'batch_size': 1024, 'dropout': 0.2, 'cell_type': 'CuDNNLSTM',
          'encoder': 'labelencoder', 'dataset': 'kdd', 'training_nb': 1,
          'resultstocsv': False, 'resultstologs': False, 'showresults': True,
          'shuffle': True}


params_var = {'encoder': ['standardscaler', 'labelencoder',
                          'minmaxscaler01', 'minmaxscaler11',
                          'ordinalencoder'],
              'optimizer': ['adam', 'sgd', 'rmsprop', 'nadam', 'adamax',
                            'adadelta'],
              'activation_fct': ['sigmoid', 'softmax', 'relu', 'tanh'],
              'layer_nb': [1, 2, 3, 4],
              'unit_nb': [4, 8, 32, 64, 128, 256],
              'dropout': [0.1, 0.2, 0.3, 0.4],
              'batch_size': [512, 1024, 2048],
          
              }

model_path = './models/'
logs_path = './logs/'
res_path = './results/' + 'testcsv/'

if params['resultstologs'] is True:
    res_name = str(params['train_data']) + '_' + str(params['features_nb']) +\
        '_' + params['loss_fct'] + '_' + params['optimizer'] + '_' +\
        params['activation_fct'] + '_' + str(params['layer_nb']) + '_' +\
        str(params['unit_nb']) + '_' + str(params['batch_size']) + '_' +\
        str(params['dropout']) + '_' + params['cell_type'] + '_' +\
        params['encoder'] + '_' + str(time())


def load_data():
    if params['dataset'] == 'kdd':
        x_train, x_test, y_train, y_test = kdd_encoding(params)
    elif params['dataset'] == 'unsw':
        x_train, x_test, y_train, y_test = unsw_encoding(params)

  
    x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
    x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])
    return x_train, x_test, y_train, y_test



from keras.layers import LSTM, GRU, SimpleRNN 
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model

def create_model(params):
    model = Sequential()

   
    if params['cell_type'].lower() == 'lstm':
        cell_type = LSTM
    elif params['cell_type'].lower() == 'gru':
        cell_type = GRU
    elif params['cell_type'].lower() == 'simplernn':
        cell_type = SimpleRNN
    else:
        raise ValueError("Invalid cell_type. Supported values: 'lstm', 'gru', 'simplernn'")

    model.add(cell_type(units=params['unit_nb'], input_shape=(x_train.shape[1], 1)))


    model.add(Dense(10, activation=params['activation_fct']))

  
    model.compile(optimizer=params['optimizer'], loss=params['loss_fct'])

    return model

def train_model(x_train, x_test, y_train, y_test, params):
    model = create_model(params)

    # Train the model
    history = model.fit(x_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(x_test, y_test))

    return history


def res_to_csv():
    ref_min_val_loss = 10 
    nsmall = 5  

    
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    full_res_path = res_path + 'full_results.csv'
    best_res_path = res_path + 'best_result.csv'


    results_df = pd.DataFrame(columns=csv_values)
    results_df.to_csv(full_res_path, index=False)

    best_res_df = pd.DataFrame(columns=csv_best_res)

    def fill_dataframe(df, history, epoch):
        df = df.append({'epochs': epoch,
                        'acc':  history.history['acc'][epoch],
                        'loss': history.history['loss'][epoch],
                        'val_acc': history.history['val_acc'][epoch],
                        'val_loss': history.history['val_loss'][epoch],
                        'train_data': params['train_data'],
                        'features_nb': params['features_nb'],
                        'loss_fct': params['loss_fct'],
                        'optimizer': params['optimizer'],
                        'activation_fct': params['activation_fct'],
                        'layer_nb': params['layer_nb'],
                        'unit_nb': params['unit_nb'],
                        'batch_size': params['batch_size'],
                        'dropout': params['dropout'],
                        'cell_type': params['cell_type'],
                        'encoder': params['encoder']},
                       ignore_index=True)
        return df

    
    def min_mean_val_loss(feature):
       
        df = pd.read_csv(res_path+feature+".csv", index_col=False)
        names = df[feature].unique().tolist()
        df_loss = pd.DataFrame(columns=names)

     
        for i in range(len(names)):
            df_value_loss = df.loc[df[feature] == names[i]]
            df_value_loss = df_value_loss.nsmallest(nsmall, 'val_loss')
            df_loss[names[i]] = np.array(df_value_loss['val_loss'])

        return df_loss.mean().idxmin(), df_loss.mean().min()

    for feature in params_var.keys():
        results_df.to_csv(res_path + feature + ".csv", index=False)
        save_feature_value = params[feature]

        for feature_value in params_var[feature]:
            df_value = pd.DataFrame(columns=csv_values)
            params[feature] = feature_value

            if feature == 'encoder' or feature == 'train_data':
                
                x_train, x_test, y_train, y_test = load_data()

            for _ in range(params['training_nb']):
                history = train_model(x_train, x_test, y_train, y_test)

             
                for epoch in range(params['epochs']):
                    df_value = fill_dataframe(df_value, history, epoch)
          
            df_value.to_csv(full_res_path, header=False, index=False, mode='a')
            df_value.to_csv(res_path + feature + ".csv", header=False,
                            index=False, mode='a')
        
        feature_value_min_loss, min_mean_loss = min_mean_val_loss(feature)

       
        if min_mean_loss < ref_min_val_loss:
            params[feature] = feature_value_min_loss
            ref_min_val_loss = min_mean_loss
        else:
            params[feature] = save_feature_value

        best_res_df = best_res_df.append({'param': feature,
                                          'value': params[feature],
                                          'min_mean_val_loss': min_mean_loss},
                                         ignore_index=True)
        best_res_df.to_csv(best_res_path, index=False)

        params = {
    'epochs': 10,
    'train_data': 494021,
    'features_nb': 4,
    'loss_fct': 'mse',
    'optimizer': 'rmsprop',
    'activation_fct': 'sigmoid',
    'layer_nb': 1,
    'unit_nb': 128,
    'batch_size': 1024,
    'dropout': 0.2,
    'cell_type': 'lstm', 
    'encoder': 'labelencoder',
    'dataset': 'kdd',
    'training_nb': 1,
    'resultstocsv': False,
    'resultstologs': False,
    'showresults': True,
    'shuffle': True
}


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()

    for i in range(params['training_nb']):
        if params['resultstocsv'] is False:
            train_model(x_train, x_test, y_train, y_test,params)
        else:
            res_to_csv()

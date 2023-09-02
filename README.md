Here's a README file for your code:

# Deep Learning and Machine Learning Models for Network Intrusion Detection

This repository contains Python code for training deep learning and machine learning models for network intrusion detection. The models are trained on two different datasets: KDD Cup 1999 and UNSW-NB15.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3
- TensorFlow
- Keras
- scikit-learn
- pandas
- numpy

You can install these packages using pip:

```bash
pip install tensorflow keras scikit-learn pandas numpy
```

## Usage

### Deep Learning Models (Keras)

1. Import the necessary libraries at the beginning of your script:

   ```python
   import pandas as pd
   import numpy as np
   import os
   from time import time
   from keras.layers import Dense, Dropout, CuDNNLSTM, CuDNNGRU, RNN, LSTM, GRU
   from keras import Sequential
   from keras.callbacks import TensorBoard, ModelCheckpoint
   from kdd_processing import kdd_encoding
   from unsw_processing import unsw_encoding
   from results_visualisation import print_results
   ```

2. Define the parameters for your training:

   ```python
   csv_values = ['epochs', 'acc', 'loss', 'val_acc', 'val_loss', "train_data",
                 "features_nb", 'loss_fct', 'optimizer', 'activation_fct',
                 'layer_nb', 'unit_nb', 'batch_size', 'dropout', 'cell_type',
                 'encoder']
   
   csv_best_res = ['param', 'value', 'min_mean_val_loss']
   
   params = {
       'epochs': 3,
       'train_data': 494021,
       'features_nb': 4,
       'loss_fct': 'mse',
       'optimizer': 'rmsprop',
       'activation_fct': 'sigmoid',
       'layer_nb': 1,
       'unit_nb': 128,
       'batch_size': 1024,
       'dropout': 0.2,
       'cell_type': 'CuDNNLSTM',
       'encoder': 'labelencoder',
       'dataset': 'kdd',
       'training_nb': 1,
       'resultstocsv': False,
       'resultstologs': False,
       'showresults': True,
       'shuffle': True
   }
   
   params_var = {
       'encoder': ['standardscaler', 'labelencoder', 'minmaxscaler01', 'minmaxscaler11', 'ordinalencoder'],
       'optimizer': ['adam', 'sgd', 'rmsprop', 'nadam', 'adamax', 'adadelta'],
       'activation_fct': ['sigmoid', 'softmax', 'relu', 'tanh'],
       'layer_nb': [1, 2, 3, 4],
       'unit_nb': [4, 8, 32, 64, 128, 256],
       'dropout': [0.1, 0.2, 0.3, 0.4],
       'batch_size': [512, 1024, 2048]
   }
   ```

3. Load and preprocess your data using the `load_data` function. You can choose between the 'kdd' and 'unsw' datasets:

   ```python
   x_train, x_test, y_train, y_test = load_data()
   ```

4. Train your deep learning model using the `train_model` function:

   ```python
   history = train_model(x_train, x_test, y_train, y_test)
   ```

5. If you want to save the results to a CSV file, you can use the `res_to_csv` function:

   ```python
   res_to_csv()
   ```

6. Run your script and observe the training results.

### Machine Learning Models (scikit-learn)

1. Import the necessary libraries at the beginning of your script:

   ```python
   import numpy as np
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.neural_network import MLPClassifier
   from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_score, auc)
   from kdd_processing import kdd_encoding
   from unsw_processing import unsw_encoding
   ```

2. Define the parameters for your training:

   ```python
   params = {
       'train_data': 494021,
       'features_nb': 4,
       'batch_size': 1024,
       'encoder': 'standardscaler',
       'dataset': 'kdd'
   }

   params_var = {
       'encoder': ['standardscaler', 'labelencoder', 'minmaxscaler01', 'minmaxscaler11', 'ordinalencoder'],
       'batch_size': [128, 256, 512, 1024, 2048]
   }
   ```

3. Load and preprocess your data using the `load_data` function. You can choose between the 'kdd' and 'unsw' datasets:

   ```python
   x_train, x_test, y_train, y_test = load_data()
   ```

4. Train your machine learning models using functions like `MLPClassifier_train`, `RandomForestClassifier_train`, `DecisionTreeClassifier_train`, and `KNeighborsClassifier_train`.

5. Run your script and observe the training results for each machine learning model.

## Dataset

This code supports two datasets:

- KDD Cup 1999: A dataset commonly used for intrusion detection research.
- UNSW-NB15: A newer dataset specifically designed for network intrusion detection.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This code was developed as part of a research project on network intrusion detection.
- Special thanks to the authors of the KDD Cup 1999 dataset and the UNSW-NB15 dataset for providing valuable data for research purposes.

Feel free to modify and use this code for your network intrusion detection projects. If you have any questions or issues, please don't hesitate to contact us.

Happy coding!

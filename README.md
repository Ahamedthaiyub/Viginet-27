


![Black and Red Gradient Professional LinkedIn Banner (1)](https://github.com/Ahamedthaiyub/Breach/assets/98688617/f3807dca-38ca-4ae9-9e18-c021ec1b1a60)







## Problem Statement

In an increasingly interconnected digital landscape, organizations face an escalating threat of cybersecurity breaches and attacks. The challenge lies in efficiently detecting and responding to these threats to safeguard sensitive data, critical systems, and maintain business continuity. Traditional security measures often fall short in identifying sophisticated and evolving threats, necessitating a comprehensive solution that combines continuous monitoring and advanced analytics.


## Solution 
Our project addresses this problem by developing a Viginet Breach Detection System (VBDS) with Machine Learning capabilities. This system aims to proactively detect security breaches, unauthorized access, and malicious activities within an organization's network and systems. By leveraging ML algorithms and real-time monitoring, VBDS enhances the organization's ability to identify, assess, and respond to security incidents swiftly and effectively.


## Introducing Viginet Breach Detection Systems 
In today's dynamic cybersecurity landscape, Viginet Breach Detection Systems, armed with Machine Learning capabilities, are indispensable. These systems leverage ML algorithms to continuously monitor network traffic and system logs, swiftly identifying anomalies and potential breaches. By learning from vast datasets and historical incidents, they adeptly discern legitimate activities from malicious intrusions. This proactive approach empowers organizations to fortify their defenses, respond promptly to threats, and safeguard critical assets and sensitive data in the face of increasingly sophisticated cyberattacks.



## Methodology-
![Blank diagram](https://github.com/Ahamedthaiyub/Breach/assets/98688617/9631804e-dbd2-4c6d-bcea-f033e7bb5eee)



# story of the model-

In the world of cyber threats, Viginet's VBDS-ML emerged as an unstoppable sentinel. With machine learning as its sword and real-time vigilance as its shield, it safeguarded organizations from the shadows of digital darkness, ensuring a secure and thriving digital future.


VBDS-ML, a true digital guardian, detected breaches before they could manifest, quelling threats with swift precision. Its adaptability and unwavering watchfulness made it the trusted protector of the digital realm, standing as an unyielding shield against the relentless onslaught of cyber adversaries.

# Let’s learn Intel oneAPI AI Analytics Toolkit​
# Introduction:

Within the domain of Viginet Breach Detection Systems, the optimization of code for superior performance is a paramount concern. This is where Intel One API proves invaluable. Intel One API offers a consolidated and simplified programming model engineered to accelerate the execution of Viginet's high-performance breach detection tasks. By harnessing Intel One API's capabilities, we can enhance the performance of the Viginet Breach Detection System, resulting in increased speed and efficiency.



# Features-

Our model offers the following features:
###  Real-Time Monitoring:

Continuously monitors network traffic and system logs in real-time to detect suspicious activities and potential security breaches as they happen.

### Machine Learning Algorithms:

Employs advanced ML algorithms to analyze data patterns, identifying anomalies and deviations from normal behavior, thereby enhancing breach detection accuracy.

### Behavioral Analysis: 

Utilizes behavioral analysis to establish baseline user and system behavior, effectively detecting deviations that may indicate unauthorized access or malicious activity.

### Historical Incident Learning: 

Learns from historical security incidents and adapts its detection mechanisms to stay ahead of evolving threats.

### Anomaly Detection: 

Identifies abnormal patterns, network anomalies, and unusual data transfer activities, triggering alerts for potential breaches.

## Getting Started-

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
 ## Dataset

This code supports two datasets:

- KDD Cup 1999
- UNSW-NB15






## Currently-

Our website is focused on the cybersecurity industry, with an emphasis on our Viginet Breach Detection System. To optimize our model, we have integrated the Intelex extension. Please be aware that our model is not yet ready for public release, and GitHub resources are not available at this time. This decision was made by Ahamed Thaiyub A

## Demo👇




https://github.com/Ahamedthaiyub/Breach/assets/98688617/6b3e4364-bcb5-44a3-96de-610b9fe46869



### webpage is under developement





## Output of our model-
![Screenshot (357)](https://github.com/Ahamedthaiyub/Breach/assets/98688617/0b591726-dae3-4a24-a1fa-01072a0ca8da)


this  visualizes the number of breaches over time, offering insights into breach trends within the Viginet Breach Detection System.




![Screenshot (358)](https://github.com/Ahamedthaiyub/Breach/assets/98688617/1686309b-362f-4fe8-9299-4f5b468a35b2)


this illustrates correlations between different breach types, providing a quick overview of relationships and potential patterns in breach occurrences.
## Note -
It's important that github does'nt supports cufflinks and poltly, So visuals are not there in  files. So if you wish to see it kindly clone the git files.

## Viginet, 
a true digital guardian, detected breaches before they could manifest, quelling threats with swift precision. Its adaptability and unwavering watchfulness made it the trusted protector of the digital realm, standing as an unyielding shield against the relentless onslaught of cyber adversaries.

## Contributors-

This project was developed by 

- Ahamed Thaiyub A(CSE)



---------------------------------------------------------------------------------------------------------------------------------------------------

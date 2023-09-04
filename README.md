


![Black and Red Gradient Professional LinkedIn Banner (1)](https://github.com/Ahamedthaiyub/Breach/assets/98688617/f3807dca-38ca-4ae9-9e18-c021ec1b1a60)







## Problem Statement

In an increasingly interconnected digital landscape, organizations face an escalating threat of cybersecurity breaches and attacks. The challenge lies in efficiently detecting and responding to these threats to safeguard sensitive data, critical systems, and maintain business continuity. Traditional security measures often fall short in identifying sophisticated and evolving threats, necessitating a comprehensive solution that combines continuous monitoring and advanced analytics.


## Solution 
Our project addresses this problem by developing a Viginet Breach Detection System (VBDS) with Machine Learning capabilities. This system aims to proactively detect security breaches, unauthorized access, and malicious activities within an organization's network and systems. By leveraging ML algorithms and real-time monitoring, VBDS enhances the organization's ability to identify, assess, and respond to security incidents swiftly and effectively.


## Introducing Viginet 
In today's dynamic cybersecurity landscape, Viginet Breach Detection Systems, armed with Machine Learning capabilities, are indispensable. These systems leverage ML algorithms to continuously monitor network traffic and system logs, swiftly identifying anomalies and potential breaches. By learning from vast datasets and historical incidents, they adeptly discern legitimate activities from malicious intrusions. This proactive approach empowers organizations to fortify their defenses, respond promptly to threats, and safeguard critical assets and sensitive data in the face of increasingly sophisticated cyberattacks.



## Methodology-
![Blank diagram](https://github.com/Ahamedthaiyub/Breach/assets/98688617/9631804e-dbd2-4c6d-bcea-f033e7bb5eee)



# story of the model-

In the world of cyber threats, Viginet's VBDS-ML emerged as an unstoppable sentinel. With machine learning as its sword and real-time vigilance as its shield, it safeguarded organizations from the shadows of digital darkness, ensuring a secure and thriving digital future.


VBDS-ML, a true digital guardian, detected breaches before they could manifest, quelling threats with swift precision. Its adaptability and unwavering watchfulness made it the trusted protector of the digital realm, standing as an unyielding shield against the relentless onslaught of cyber adversaries.

### Let’s learn Intel oneAPI AI Analytics Toolkit​
# Introduction:

Within the domain of Viginet Breach Detection Systems, the optimization of code for superior performance is a paramount concern. This is where Intel One API proves invaluable. Intel One API offers a consolidated and simplified programming model engineered to accelerate the execution of Viginet's high-performance breach detection tasks. By harnessing Intel One API's capabilities, we can enhance the performance of the Viginet Breach Detection System, resulting in increased speed and efficiency.

Intel has developed optimized AI software products aimed at enhancing performance in AI workloads. One of these products is the AI Kit, designed to deliver improved end-to-end performance for AI tasks. Additionally, Intel has introduced the Intel Extension for Scikit-learn, which accelerates training models, leading to more efficient and effective models.

This acceleration in model training has a significant impact on network intrusion detection. It results in faster detection speeds and improved accuracy in classifying threat categories. Ultimately, these enhancements enable enterprises to gain threat insights more quickly, allowing them to take rapid action in response to security threats.

## Intel® AI Software Products:

Intel® AI Analytics Toolkit (AI Kit)
Intel® Extension for Scikit-learn*



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


scikit-learn-intelex
geopy==2.3.0
numpy==1.22.2
pandas==1.4.1
spacy==2.3.5
scikit-learn==0.15.2 
Python==3.9
tensorflow==2.13.0

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
   from processing import kdd_encoding
   from unsw import unsw_encoding
   from result import print_results
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




https://github.com/Ahamedthaiyub/Viginet-BreachGuard/assets/98688617/d42e2825-c966-4cbd-b54a-0679edb4e070





### webpage is under developement
# Documentation
## VigiNet: Intrusion Detection with Recurrent Neural Networks (RNN)

**Project Overview:**

This documentation provides an overview of the codebase for VigiNet, an Intrusion Detection System (IDS) using Recurrent Neural Networks (RNNs). The code is structured into two main files: `st.py` for the Streamlit application and `train.py` for training the RNN model.

**Table of Contents:**

1. Introduction
2. File Structure
3. Usage
4. Data Loading and Preprocessing
5. Model Architecture
6. Training the Model
7. Streamlit Application
8. Conclusion

## 1. Introduction

VigiNet is  designed to enhance network security through the power of Recurrent Neural Networks (RNNs). This project leverages the Intrusion Detection AI Reference Kit from Intel for optimized performance.

## 2. File Structure

- **st.py**: Contains the Streamlit application for interacting with VigiNet.
- **train.py**: Contains code for data loading, preprocessing, model creation, and training.

## 3. Usage


### Running the Application

1. Open a terminal.
2. Navigate to the project directory.
3. Execute `streamlit run st.py`.
4. Execute `python app.py`

## 4. Data Loading and Preprocessing

- Data is loaded and preprocessed in the `load_data()` function in `train.py`.
- Ensure your dataset is appropriately prepared and loaded within this function.

## 5. Model Architecture

- The RNN model architecture is created in the `create_model(params)` function in `train.py`.
- The model type (`LSTM`, `GRU`, or `SimpleRNN`) is determined based on the `cell_type` parameter.
- Modify the model layers and structure as needed.

## 6. Training the Model

- The training process is handled in the `train_model()` function in `train.py`.
- Adjust training parameters in the `params` dictionary in `st.py`.
- Execute training by clicking the "Start Training" button in the Streamlit application.

## 7. Streamlit Application

- The Streamlit application (`st.py`) provides an interactive interface for starting the training process.
- Visualizations and results can be added to this file based on your requirements.

## 8. Conclusion

VigiNet, powered by RNNs and optimized with Intel's Intrusion Detection AI Reference Kit, serves as a robust foundation for building an IDS. It includes data loading, model creation, training, and a Streamlit interface for ease of use. Customize and enhance it to meet your specific project needs.

*Note: VigiNet is not only an IDS but also a showcase of Intel's AI reference kit for intrusion detection, delivering optimized performance with Intel OneAPI.*




This documentation summarizes the key aspects of the codebase. For more detailed explanations, consult the code and relevant comments within each file.

Please note that the actual code files should include detailed comments and documentation within the code to facilitate further understanding and maintenance.







## Output of our model-
![Screenshot (357)](https://github.com/Ahamedthaiyub/Breach/assets/98688617/0b591726-dae3-4a24-a1fa-01072a0ca8da)


this  visualizes the number of breaches over time, offering insights into breach trends within the Viginet Breach Detection System.




![Screenshot (358)](https://github.com/Ahamedthaiyub/Breach/assets/98688617/1686309b-362f-4fe8-9299-4f5b468a35b2)


this illustrates correlations between different breach types, providing a quick overview of relationships and potential patterns in breach occurrences.
## Note -
It's important that github does'nt supports cufflinks and poltly, So visuals are not there in  files. 

## Viginet
a true digital guardian, detected breaches before they could manifest, quelling threats with swift precision. Its adaptability and unwavering watchfulness made it the trusted protector of the digital realm, standing as an unyielding shield against the relentless onslaught of cyber adversaries.

## Contributors-

This project was developed by || Ahamed Thaiyub A❤️



---------------------------------------------------------------------------------------------------------------------------------------------------

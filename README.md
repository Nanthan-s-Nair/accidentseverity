Accident Severity Prediction Neural Network
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Table of Contents

Overview
Prerequisites
Usage
Load and Preprocess Data
Encode Features
Encode Target Variable
Split Data and Normalize Features
Build Neural Network Model
Compile and Train the Model
Evaluate and Predict
Decode Predictions and Display Results
Print Results
Model Flexibility
Output Explanation

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Overview

  This repository contains a simple implementation of a neural network for predicting accident severity based on a dataset related to traffic accidents. The neural network is built using     the Keras library and trained on features such as location, weather, light conditions, road type, vehicle type, and driver behavior.
  
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Prerequisites

  Before running the code, make sure you have the necessary libraries installed.
  
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Usage

Load and Preprocess Data
  The dataset is loaded from the 'testml2 - Sheet1 (1).csv' file. Rows with missing values are dropped, and irrelevant columns ('Accident_Severity', 'Date') are excluded.

Encode Features
  Categorical features ('Location', 'Weather', 'Light_Condition', 'Road_Type', 'Vehicle_Type', 'Driver_Behavior') are encoded using Label Encoding.

Encode Target Variable
  'Accident_Severity' is encoded into numerical values.

Split Data and Normalize Features
  The data is split into training and testing sets. Features are normalized using StandardScaler.

Build Neural Network Model
  A neural network model with two hidden layers and an output layer is defined.

Compile and Train the Model
  The model is compiled with the Adam optimizer and sparse categorical crossentropy loss. It is trained on the training set with early stopping to prevent overfitting.

Evaluate and Predict
  The model is evaluated on the test set. Predictions are made on the test set.

Decode Predictions and Display Results
  Predictions are decoded back to their original labels. A DataFrame is created to compare actual and predicted accident severity.

Print Results
  Test loss, test accuracy, and the DataFrame comparing actual and predicted values are printed.
  
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Model Flexibility
  The neural network model implemented in this repository exhibits a certain level of flexibility, which can be attributed to several factors:

Architecture Flexibility
  The model architecture can be easily modified to adapt to different data characteristics and modeling requirements. Users can adjust the number of hidden layers, the number of neurons in   each layer, and the activation functions. For example, if the dataset is complex, users might choose to increase the network's depth or width to capture intricate patterns.

Hyperparameter Tuning
  The code includes options for hyperparameter tuning, such as adjusting the learning rate, batch size, and the number of epochs. Users can experiment with different hyperparameter       
  configurations to find the optimal settings for their specific dataset.

Early Stopping
  The implementation incorporates early stopping, a technique that monitors the model's performance on a validation set and halts training when it no longer improves. This prevents         
  overfitting and enhances the model's generalization to new data.

Dataset Adaptability
  The model is designed to handle categorical features through label encoding. Users can easily adapt the encoding process to suit their specific dataset, including handling different        categorical columns or employing alternative encoding techniques.
  
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Output Explanation

  The output of the code indicates the training progress over multiple epochs. The model achieves perfect accuracy (1.0000) on the test set, suggesting successful learning of patterns in     the data. The DataFrame results displays the actual and predicted accident severity for each sample in the test set.This model basically aims in flexibility and helps the fellow user to    further adapt this model on to various data sets by changing the keys in the table and further altering the code fo changing required columns.
  
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


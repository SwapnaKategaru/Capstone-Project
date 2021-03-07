# Capstone Project - Azure Machine Learning Engineer

This project is part of the Udacity Azure ML Nanodegree. In this project, we build a machine learning model using the Python SDK and a provided Scikit-learn model. In this project, we create two models - one using Automated ML and one customized model whose hyperparameters are tuned using HyperDrive. The best performing model is deployed as a web service.

## Dataset

The dataset used in this notebook is heart_failure_clinical_records_dataset.csv which is an external dataset available in kaggle.
This dataset contains data of 299 patients and 12 features that are useful to predict mortality by heart failure.


### Overview
No.of patients data collected : 299
Input variables or features : age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time
Output/target variable : DEATH_EVENT

### Task
In this project, we create a classification model for predicting mortality rate/DEATH_EVENT(target variabe) that is caused due to Heart Failure.

### Access
* Importing of data from csv file (heart_failure_clinical_records_dataset) using azureml's TabularDatasetFactory class.
* Create a tabular dataset.

## Automated ML

`Automl Settings` -  Using AUC weighted as the primary metric, featurisation set to auto, max_concurrent_iterations set as '4' for maximum number of iterations to execute in parallel, verbosity level set to default as logging.INFO for writing to log file.    
`AutoML Configuration` - Classification experiment with experiment timeout minutes set to 15 minutes and 2 cross-validation folds. Blocked model is XGBoostClassifier along with values for training data and label column name.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The best performing model was AutoML with accuracy of 0.9154738177206015 by Voting Ensemble algorithm.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

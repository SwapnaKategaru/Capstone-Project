# Capstone Project - Azure Machine Learning Engineer

This project is part of the Udacity Azure ML Nanodegree. In this project, we build a machine learning model using the Python SDK and a provided Scikit-learn model. In this project, we create two models - one using Automated ML and one customized model whose hyperparameters are tuned using HyperDrive. The best performing model is deployed as a web service.

## Dataset

The dataset used in this notebook is heart_failure_clinical_records_dataset.csv which is an external dataset available in kaggle.
This dataset contains data of 299 patients and 12 features that are useful to predict mortality by heart failure.


### Overview
**No.of patients data collected** : 299   
**Input variables or features** : age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time   
**Output/target variable** : DEATH_EVENT

### Task
In this project, we create a classification model for predicting mortality rate/DEATH_EVENT(target variabe) that is caused due to Heart Failure.

### Access
* Importing of data from csv file (heart_failure_clinical_records_dataset) using azureml's TabularDatasetFactory class.
* Create a tabular dataset.

## Automated ML

`Automl Settings` -  Using AUC weighted as the primary metric, featurisation set to auto, max_concurrent_iterations set as '4' for maximum number of iterations to execute in parallel, verbosity level set to default as logging.INFO for writing to log file.    
`AutoML Configuration` - Classification experiment with experiment timeout minutes set to 15 minutes and 2 cross-validation folds. Blocked model is XGBoostClassifier along with values for training data and label column name parameters.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

Voting Ensemble algorithm is generated by AutoML as the best model and the best metric score is **0.9**. Ensembled algorithms of this model includes 'GradientBoosting', 'RandomForest' and 'LightGBM'. Best ensemble weights for these algorithms is 0.1 and best individual pipeline score is 0.885.

A prefitted soft voting classifier is applied where every individual classifier provides a probability value, the predictions are weighted according to classifier's importance that are summed up and greatest sum of weighted probabilities wins the vote. 

Parameters of gradientboostingclassifier are:

* __learning_rate__ : Learning rate shrinks the contribution of each tree
* __max_depth__ : Maximum depth limits the number of nodes in the tree
* __loss__ : Loss function(deviance) to optimize for classification with probabilistic outputs

Parameters of randomforestclassifier are:

* __max_features__ : Number of features to consider when looking for the best split
* __min_samples_leaf__ : Minimum number of samples required to be at a leaf node
* __n_jobs__ : Number of jobs to run in parallel

Parameters of lightgbmclassifier are:

* __importance_type__ : Type of feature importance to fill into feature importance includes split and gain
* __n_estimators__ : Number of boosted trees to fit having default set to 100
* __num_leaves__ : Maximum tree leaves for base learners

This model can be improved further by specifying additional parameters for automl configuration and settings that contribute for its better performance. Some of the ways to improve could be using appropriate no.of cross validations as it reduces bias and improves generalizing pattern by the model, specifying custom ensemble behavior in an AutoMLConfig object using ensemble setting of parameters. Also, the xgboostclassifier is blocked in automl config due to incompatible dependency issue for sdk version(1.22.0) used for this project and enabling xgboost can improve performance as it uses a more regularized model formalization that controls over-fitting.


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

The model used for this experiment is Logistic Regression. It is because logistic regression is simple and effective classification algorithm used for binary classification task as seen in this case we are predicting the DEATH_EVENT variable of an individual and gives good accuracy for simple datasets. Logistic regression uses maximum likelihood estimation (MLE) to obtain the model coefficients that relate predictors to the target.

### Types of parameters:

Configuration for hyperdrive run to execute experiment with specified parameters like maximum total no.of runs to create and execute concurrently, name of the primary metric and primary metric goal is defined along with following hyperparameters:

**Parameter sampler :**   

Specifying parameter sampler using *RandomParameterSampling* class that enables random sampling over a hyperparameter search space from a set of discrete or continuous values(*C* and *max_iter*). 

Random Parameter Sampler is used as:

* *Random parameter sampling* supports both discrete and continuous hyperparameter values.
* Helps to identify low performing runs and thereby helps in early termination.
* Low bias in random sampling as hyperparameter values are randomly selected from the defined search space and have equal probability of being selected.
* *choice* function helps to sample from only specified set of values.
* *uniform* function helps to maintain uniform distribution of samples taken.

**Policy :**    

Specifies early termintaion policy for early stopping with required amount of *evaluation interval*, *slack factor* and *delay_evaluation*.

Bandit Policy is used as:

* It is a early termination policy that terminates low performing runs which improves computation for existing runs.
* It terminates runs if primary metric is not within specified slack factor in comparison to the best performing run.
* This policy is based on slack factor and evaluation interval.


### Results

*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
The hyperdrive optimized logistic regression model produced an accuracy of 0.72. Parameters of the model includes:

* __C__ : Inverse of regularization strength where smaller values specify stronger regularization
* __max_iter__ : Maximum number of iterations taken for the solvers to converge
* __penalty__ : Used to specify the norm used in the penalization
* __fit_intercept__ : Specifies if a constant(bias or intercept) should be added to the decision function
* __intercept_scaling__ : A synthetic feature with constant value equal to intercept_scaling appended to instance vector
* __tol__ : Tolerance for stopping criteria

Hyperdrive model can be improved further by tuning with different hyperparameters parameters that contribute for improvement in its performance. We can also improve the scoring by optimizing with other metrics like Log Loss and F1-Score. Use more appropriate parameters for hyperdrive configuration settings and increase the count of maximum total runs and concurrent runs. 


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input. 

The model with best accuracy is deployed as web service which is a **AutoML model** with better performance showing Accuracy of 0.9. 

* A model is deployed to Azure Container Instances(ACI), by creating a deployment configuration that describes the compute resources required(like number of cores and memory). 
* Creating an inference configuration, which describes the environment needed to host the model and web service.
* AzureML-AutoML environment is used which is a curated environment available in Azure Machine Learning workspace.

* Deploying an Azure Machine Learning model as a web service creates a REST API endpoint. This project shows key based authentication used and Swagger URI that is generated through inference schema in score.py script file.
* This endpoint can be used to consume web service using scoring endpoint URL and Primary Key. We can send data and make a request to this endpoint and receive the prediction returned by the model. 
* In this project, the data is requested to endpoint through endpoint.py script that has sample input from the dataset used and also using Python SDK where input json payload of two sets of data instaces is used.
* A post request is sent to endpoint that uses scoring_uri and primary key for authentication. This results in displaying json response as output from endpoint.

## Future improvements for project

* Enable deep learning while specifying classification task type for autoML as it applies default techniques depending on the number of rows present in training dataset provided and applies train/validation split with required no.of cross validations without explicitly being provided.
* Use iterations parameter of AutoMLConfig Class that enables use of different algorithms and parameter combinations to test during an automated ML experiment and increase experiment timeout minutes.
* Use Azure Kubernetes Services(AKS) instead of Azure Container Insance(ACI) as AKS helps in minimizing infrastructure maintenance, uses automated upgrades, repairs, monitoring and scaling. This leads to faster development and integration.
* Use Dedicated virtual machine instead of low-priority as these vm do not guarantee compute nodes.
* GPU can be used instead of CPU as it enormously increases the speed.


## Screen Recording
Link to screencast : XXXXX
Screencast demonstrates a working model, deployed model, a sample request sent to the endpoint and its response and additional feature of the model that shows enabling application insights through logs.py script file. Logging the deployed model is important as it helps detect anomalies, includes analytic tools, retrieve logs from a deployed model and has vital role to debug problems in production environments.


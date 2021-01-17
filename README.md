# Heart Failure Prediction with MICROSOFT AZURE

In this project, I will be creating two models:one using Automated ML one customized model whose hyperparameters are tuned using HyperDrive for predicting death event of a patient.I had compared the performance of both the models and deploy the best performing model.
 
 ## Project Workflow
 
 ![Screenshot (443)](https://user-images.githubusercontent.com/75804779/104836617-219abf00-58d5-11eb-89e0-affd20ac6478.png)

## Project Set Up and Installation
The starter files needed to run this project are the following:

automl.ipynb : Jupyter Notebook to run the autoML experiment

hyperparameter_tuning.ipynb : Jupyter Notebook to run the Hyperdrive experiment

train.py : Script used in Hyperdrive

score.py : Script used to deploy the model

heart_failure_clinical_records_dataset.csv : The dataset

## Dataset

### Overview
This dataset can be found in https://www.kaggle.com/andrewmvd/heart-failure-clinical-data. Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

1.Age

2.Anaemia:Decrease of red blood cells or hemoglobin (boolean)

3.Creatinine_phosphokinase:Level of the CPK enzyme in the blood (mcg/L)

4.Diabetes:If the patient has diabetes (boolean)

5.Ejection_fraction:Percentage of blood leaving the heart at each contraction

6.High_blood_pressure:If the patient has hypertension (boolean)

7.Platelets:Platelets in the blood (kiloplatelets/mL)

8.Serum_creatinine:Level of serum creatinine in the blood (mg/dL)

9.Serum_sodium:Level of serum sodium in the blood (mEq/L)

10.Sex:Woman or man (binary)

11.Smoking

12.Time

### Task
To predict whether the persion will get have heart failure or not.

### Access
We use 2 ways to access the data in the workspace:

1.In AutoML I used read_csv() Pandas function to get file locally.

2.For Hyperdrive, I used Dataset.Tabular.from_delimited_files() in the train script to get the file with URL.

## Automated ML
Following are the automl sttings and configuration for this experiment.
![Screenshot (446)](https://user-images.githubusercontent.com/75804779/104837000-c4ecd380-58d7-11eb-8094-63391e66f104.png)

Important automl settings that I used are experiment_timeout_minutes and max_concurrent_iterations.The experiment_timeout_minutes means the maximum amount of time in minutes that all iterations combined can take before the experiment terminates and the max_concurrent_iterations means the maximun number of iterations that would be executed in parallel.

Here I have selected task Classification, with AUC_weighted as primary metric and the target I want to find is the  DEATH_EVENT. The train_data is0 TabularDataset type.Finally, I have enabled early stopping to avoid overfitting.

### Results
I got RunDetails as follows:

![Screenshot (449)](https://user-images.githubusercontent.com/75804779/104837240-34af8e00-58d9-11eb-9627-a578ff8abd25.png)
![Screenshot (450)](https://user-images.githubusercontent.com/75804779/104837247-44c76d80-58d9-11eb-8359-8b8f4c1e8f0a.png)
![Screenshot (451)](https://user-images.githubusercontent.com/75804779/104837270-67f21d00-58d9-11eb-9bcd-f1727f139789.png)

We can see above that the Dataset pass the 4 Data Guardrails: Cross Validation, Class Balancing Detection, Missing Feature Value and High Cardinality Feature Detection.

The best model was VotingEnsemble with 0.8665.The Voting Ensemble estimates multiple base models and uses voting to combine the individual predictions to arrive at the final ones.

The best model with its run Id is shown below.
![Screenshot (452)](https://user-images.githubusercontent.com/75804779/104837346-dcc55700-58d9-11eb-8251-eaba9002ab18.png)

The parameters of best model trained are:
![Screenshot (454)](https://user-images.githubusercontent.com/75804779/104837416-50fffa80-58da-11eb-8987-005af0e1c452.png)

## Hyperparameter Tuning
Hyperparameter tuning is done by building a hyperdrive service using Jupyter notebook.First I initialize the azure machine learning workspace, then created a compute cluster to run the experiments on and check for existing cluster. Now the existing cluster is found so it was used instead of creating a new cluster.For this Logistic Regression algorithm was used.This is based on hyperparameters such as -C(Inverse of Regularization Strength) and -max_iter(Maximum number of iterations to converge). ps = RandomParameterSampling({ "--C" : uniform(0.5,1.0), "--max_iter" : choice(50,100,150,200) })

The sampling method I used is RandomSampling.It supports early termination of low-performance runs.For this a scikit-learn estimator for the training script and a HyperDriveConfig was created.

The advantage of using Random Sampling is that it helps to avoid bias.It also helps in choosing the best hyperparameters and optimize for speed versus accuracy. It supports both discrete and continuous values. It supports early termination of low-performance runs. In Random Sampling, the values are selected randomly from a defined search space.

With the help of Early Termination policy, we can terminate poorly performing runs.Here I used Bandit Policy.Bandit Policy is based on slack factor and evaluation interval. This policy will terminate runs whose primary metric is not within the specified slack factor.The bandit policy helped to avoid burning up a lot of resources while trying to find an optimal parameter, it terminates any run that does not fall within the slack factor's range.

### Results
The RunDetails are as follows:
![Screenshot (457)](https://user-images.githubusercontent.com/75804779/104840131-7f3a0600-58eb-11eb-990a-822b3228b317.png)

The best run, its run id and hyperparameters are shown below
![Screenshot (459)](https://user-images.githubusercontent.com/75804779/104840904-3e8ebc80-58ec-11eb-852b-3db8342d3939.png)

![Screenshot (460)](https://user-images.githubusercontent.com/75804779/104841470-667e2000-58ec-11eb-8dac-21bfe4eb8b0b.png)

## Model Deployment
In AutoML model I got the accuracy 0.8665 and in hyperparameter tuning by hyperdrive I got accuracy 0.85.
So,the best model was of AutoML and I decided to deploy it.

Following steps are performed to deploy the model:
1.Creating a scoring script for sending request to the web service. This script must have the init() and run(input_data) functions.

2.Defining the inference and the deployment configuration

3.Create the environment for the deployment where Azure Machine Learning can install the necessary packages

![Screenshot (462)](https://user-images.githubusercontent.com/75804779/104842940-2324b100-58ee-11eb-934d-e3b4e072b38b.png)

![Screenshot (464)](https://user-images.githubusercontent.com/75804779/104843275-c32f0a00-58ef-11eb-9499-c128b1811e59.png)
In this we can se that the ststus is termed as 'Healthy'.
## Screen Recording

Link: https://drive.google.com/file/d/1LGBoCP8YHhU1SB9cnFeKKa1NY1Bxbm_z/view?usp=sharing

## Standout Suggestions
1.Can convert  the model to ONNX format.
2.We can use of other classification algorithms in hyperdrive run.
3.Better tuning of hyperparameters can be done.We can improve the Parameter sampler.

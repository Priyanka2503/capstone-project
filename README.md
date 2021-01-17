# Heart Failure Prediction with MICROSOFT AZURE

 In this project, I will be creating two models: one using Automated ML one customized model whose hyperparameters are tuned using HyperDrive for predicting death event of a patient.I had compared the performance of both the models and deploy the best performing model.
 
 ## Project Workflow
 
 ![Screenshot (443)](https://user-images.githubusercontent.com/75804779/104836617-219abf00-58d5-11eb-89e0-affd20ac6478.png)

## Project Set Up and Installation
The starter files needed to run this project are the following:

automl.ipynb:Jupyter Notebook to run the autoML experiment

hyperparameter_tuning.ipynb:Jupyter Notebook to run the Hyperdrive experiment

train.py:Script used in Hyperdrive

score.py:Script used to deploy the model

heart_failure_clinical_records_dataset.csv:The dataset

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

The best model was VotingEnsemble with 0.8665.

The best model with its run Id is shown below.
![Screenshot (452)](https://user-images.githubusercontent.com/75804779/104837346-dcc55700-58d9-11eb-8251-eaba9002ab18.png)

The parameters of best model trained are:
![Screenshot (454)](https://user-images.githubusercontent.com/75804779/104837416-50fffa80-58da-11eb-8987-005af0e1c452.png)

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

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

# Home Credit Group

## Introduction

In today word banks often deals with challenges in assessing the risk of loan defaults. To help them make better decisions, our startup aims to provide Risk Evaluation as a Service using machine learning. By analyzing patterns in data, we can help bank to predict risk and manage it more efficiently. 

Objectives for this Part
<li>Translating business requirements into data science tasks.</li>
<li>Perform EDA.</li>
<li>Apply statistical inference procedures.</li>
<li>Use machine learning to solve business problems.</li>
<li>Deploy machine learning model.</li>

Aim of the Project

The aim is to build a machine learning model which helps bank evaluate the loan risk. Using the Home Credit dataset, we will create models that predict which loans are likely to be risky, providing useful insights for banks. This project will include data analysis, model building and deployment to a cloud platform.

Our work will be split into separate notebooks each covering different parts of the project.

Main dataset can be downloaded from <b>[here](https://storage.googleapis.com/341-home-credit-default/home-credit-default-risk.zip)</b>.

## Project structure 
Main project divided into 2 parts:

EDA 
<br>
Feature Engineering And Model

<br>
<b>Exploratory Data Analysis:</b>
<br>
Consists of analysis of the data we have, mainly focusing on application dataset.
<br>

<b>Feature Engineering And Model</b> - will have most of the backend:
<br>
Reduce memory consumption of datasets by converting from csv files to parquet.
<br>
Lower feature data types where needed.
<br>
Add additional features, which was created base by EDA. 
<br>
Featuretools help to combine all data from all datasets we have.
<br>
Remove duplicate values, high missing value features, correlated features.
<br>
Use ML model to select only important features which help to predict TARGET.
<br>
Created separate ML models which give us baseline.
<br>
Improve ML model by using hyperparameter tuning.
<br>
Check Feature importance for models.
<br>
Build and export the best ML model base by AUC curve.


## Requirements for the project

To install all necessary libraries use `pip install -r requirements.txt`
To install only app libraries navigate to app folder and use `pip install -r requirements.txt`


## Launch ML model locally
Install Docker and Java

In terminal navigate to you `project/app` folder
`cd path_to_project/app`

Build Docker image
`docker build -t home_credit_group_app .`

Run container
`docker run -p 8000:8000 home_credit_group_app`

Or if you want to run container with docker platform
`docker run -d -p 8000:8000 home_credit_group_app`

Main page
`http://localhost:8000/`


To check running containers
`docker ps`

To stop docker container
`docker stop 'container_id'`

To predict loan status use form on home page.
Fill required field and submit the form.

If input values incorrect result will not be given. 


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact Information
[Email](ricardas.poskrebysev@gmail.com)
[LinkedIn](https://www.linkedin.com/in/ri%C4%8Dardas-poskreby%C5%A1evas-665207206/)
[GitHub](https://github.com/Riciokzz)
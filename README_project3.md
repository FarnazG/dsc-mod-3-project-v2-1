
# Module 3 Final Project


## Introduction

Project Title: Costumer Churn Data
Build a classifier to predict whether a customer will ("soon") stop doing business with SyriaTel, a telecommunications company. Note that this is a binary classification problem.


## Outline:

1.Data cleaning and preprocessing

2.Modeling

3.Evaluating and choosing the final model

4.Recommendations

5.Future works


## Project breakdown

### 1. Data cleaning and preprocessing:

* Importing libraries and available data files and Ckecking for:
* Missing data and placeholders
* Data types
* Multicollinearity
* Duplicateds and outliers
* Data distributions
* Data range and scalling
* Categorcal data 

In this project, after preprocessing steps, 4 differently preprocessed data was prepared:

data = original dataframe after:
* dropping unnecessary features
* features with multicollinearity 
* binarizing categorical data
* removing outliers

data1 = a copy of the original dataframe data in which:
* log transfer was applied on skewed distributions.
  
data2 = a copy of the original dataframe data in which:
* features were scaled

data3 = a copy of the original dataframe data in which:
* features were log-transfered and scaled.


### 2. Modeling 

* Separating the feaures and target columns in the dataset
* Splitting train and test data
* Working on rebalancing the class imbalancement for each dataset

![alt text](https://github.com/FarnazG/dsc-mod-3-project-v2-1/blob/master/project_3_images/class_imbalancement.png)

* Building the basic model(s)(Logistic Regression) for each dataset
* Creating confusion matrix and obtaining classification report for each dataset
* Creating ROC curve and AUC for each dataset

At this step we compared and pick the best dataset to work with

![alt text](https://github.com/FarnazG/dsc-mod-3-project-v2-1/blob/master/project_3_images/ROC_curve.png)

* Hyperparameter tuning
* Feature selection

![alt text](https://github.com/FarnazG/dsc-mod-3-project-v2-1/blob/master/project_3_images/feature_importance.png)



### 3. Evaluating models and choosing the final model 

* Exploring a few models considering different features and hyperparameters
* Visualizing scatter plots of each feature with the probabilities of leaving

![alt text](https://github.com/FarnazG/dsc-mod-3-project-v2-1/blob/master/project_3_images/customer_service_calls.png)

* Choosing the best model
* Evaluating the financial impact of the final model prediction on the SyriaTel company


### 4. Recommendations for the company

By default, we assume we dont know the budget of company to distribute promotion and offers and their ability to change any charge rates, so we only suggest our formula based on the raw input.

The policy can be offering promo plans to potential leaving customers to motivates them to stay.

Promo_Plan to keep customer motivated = [(potential yearly promo) + (12* potential monthly discount)]

So, to pick the optimal model based on the confusion matrix info, the model should maximize the benefit equation:

**Benefit equation:** 

TP*(monthly_contract*12)-TN*(monthly_contract*12)-FP*(monthly_contract*12)+FN*[(monthly_contract*12)-(promo_plan)] 
So the predicting model will best benefit the company if limits the false positive predictions. 


**Final model:**

* most important features: ['total_day_calls','international_plan','customer_service_calls']

* Logistic regression with tuned Hyperparameters:

```javascript
{
    'C': 0.1, 
    'class_weight': 'balanced',  
    'penalty': 'l1', 
    'random_state': 10, 
    'solver': 'liblinear'
} 
```
* Model Test Data Precision: 0.345
* Model Test Data Accuracy: 0.738 
* Model Test Data AUC: 0.845
  
![alt text](https://github.com/FarnazG/dsc-mod-3-project-v2-1/blob/master/project_3_images/confusion_matrix.png)


### 5. Future Work:

* Testing other classification algorithms and compare the results to our existing model. Decision Trees, Random     Forests, and Support Vector Machines are a few other classifiers to consider testing.

* Defining a function to show the affect of location and each specific state on the leaving probability when all other features are the same.

* Identifying new promotion plans to higher the benefit and lower the probability of leaving the company.

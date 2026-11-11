BINARY CLASSIFICATION WITH A BANK DATASET (PLAYGROUND S5E8)
Competition link: https://www.kaggle.com/competitions/playground-series-s5e8. 

Target: `y`
 (1 = client subscribes to a term deposit)
 (0 = client does not subscribe to a term deposit).
Evaluation Metric: ROC-AUC (Receiver Operating Characteristic - Area Under the Curve).

 Problem and Data

The aim is to predict whether a client will sign up for a bank term deposit, based on marketing campaign data. 
Dataset size
750,000 training rows and 250,000 test rows.
Features
17 features such as client demographics (age, job, marital status), account information (balance, housing loan), and campaign details (contact duration, month, previous outcome). 
The Metric
ROC-AUC: It measures the ability of the model to differentiate the positive class (subscribes) from the negative class (does not subscribe). A perfect score is 1.0 while random guessing is 0.5. Since only about 12% of clients subscribe, using accuracy would be misleading, therefore ROC-AUC is the correct metric to evaluate our model.

Exploratory Data Analysis (EDA) Findings

Target Population
The dataset is highly imbalanced. 87.9% of clients did not subscribe (0), while only 12.1% subscribed (1).
Categorical Relations
We found that the strongest predictor is `poutcome`, as seen on our bar charts. When the response to a client's previous campaign is "success," their subscription rate increased to nearly 80%. 
Numeric Distribution
Some Numeric Distributions are highly skewed to the right, such as `balance`, `duration`, and `campaign`. 
Missing Values and Anomalies
There are no explicit null values in the data. However, we found something odd. Although 672,434 clients have `pdays = -1` which in the data set documentation means they were never previously contacted. 

Preprocessing and Feature Engineering 

Based on the EDA, we performed the following steps:
In terms of anomaly handling, we added a new column, `pdays_flag`, (1 if the person has not been contacted, else 0), and we changed the value of the column `pdays_clean` from -1 to NaN so the model does not treat it as a mathematical number.
Log Transformation
Skewed feature was applied to the `duration` feature with a log transformation to aid the model to learn better.
Categorical Encoding: We converted text columns to pandas' category dtype and passed them through the model without using one-hot encoding which would take up heavy memory consumption.

The validation scheme and models.

To prevent overfitting and to ensure generalization of our model, we used 5-Fold Stratified Cross-Validation.
The baseline model correctly predicts "0" (the majority class) with a baseline ROC-AUC of 0.50000.
Model
We trained a LightGBM gradient boosting model.

Results: 
  Fold 1: 0.96881
  Fold 2: 0.96785
  Fold 3: 0.96783
  Fold 4: 0.96891
  Fold 5: 0.96816
  Overall OOF ROC-AUC: 0.96831.
 We are yet to train the addittional models. we will do that later but soo far this is what we have done.

 We have aso made our first submission on kaggle and this is our score 0.96905

Team Members and Roles

Joyce
 Data loading, Exploratory Data Analysis (EDA), visualizations and writing the data dictionary.
Trustina
Feature Engineering, categorical encoding, anomaly handling, and Preprocessing.
Rhoda
Model training (LightGBM), validation scheme (5-Fold CV), hyperparameter tuning and final submission generation.
Rhoda
Currently working on the hyperpararameter tuning and final ensemble.
Trustina
I am currently reviewing the feature engineering code and testing XGBoost.


































References
Course lecturer. (2025). Titanic: Exploring the data and making a first submission [Colab notebook]. [Unpublished course material].

 Kaggle. (2025a). Predicting classification of bank term deposits [Notebook].
 
 Kaggle. (2025b). Stack vs blend [Notebook].

Moro, S., Cortez, P., & Rita, P. (2014). An applied study of the prediction of the success of the bank telemarketing by using data-driven approach. *Decision Support Systems, 62*, 22–31.


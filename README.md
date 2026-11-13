BINARY CLASSIFICATION WITH A BANK DATASET (PLAYGROUND S5E8)
Competition link: https://www.kaggle.com/competitions/playground-series-s5e8. 

Target: `y`
(1 = client subscribes to a term deposit)
(0 = client does not subscribe to a term deposit).
Evaluation Metric: ROC-AUC (Receiver Operating Characteristic - Area Under the Curve).

**Problem and Data**

The aim is to predict whether or not a client will sign up for a bank term deposit, given marketing campaign data. 
Dataset size
It was 750,000 training rows and 250,000 test rows.
Features
17 features such as client demographics (age, job, marital status), account information (balance, housing loan), and campaign details (contact duration, month, previous outcome). 
The Metric
ROC-AUC: Measures the ability of the model to differentiate the positive class (subscribes) from the negative class (does not subscribe). The score is 1.0 with random guessing being 0.5. Since 12% of clients subscribe, the accuracy would be misleading, therefore it is correct to use ROC-AUC. (Moro et al., 2014).

**Exploratory Data Analysis (EDA) Findings**

*Target Population*
The dataset is highly imbalanced. 87.9% of clients did not subscribe (0), while only 12.1% subscribed (1).
*Categorical Relations*
We found that the strongest predictor is `poutcome`, as seen on our bar charts. When the response to a client's previous campaign is "success," their rate of subscribing to the campaign increases almost to 80%. 
*Numeric Distribution*
Some Numeric Distributions are highly skewed to the right, such as `balance`, `duration`, and `campaign`. 
*Missing Values and Anomalies*
There is no explicit null values in the data. We found something odd though: 672,434 clients have `pdays = -1` which in the data set documentation means they were never previously contacted. 

**Preprocessing and Feature Engineering**

Based on the EDA we conducted the following:
In terms of anomaly handling, we added a new column, `pdays_flag`, (1 if the person has not been contacted, else 0), and we changed the value of the column `pdays_clean` from -1 to NaN so that the model does not treat it as a mathematical number.
*Log Transformation*
Skewed feature was applied to the `duration` feature with a log transformation to aid the model to learn better. (Course lecturer, 2025).
*Categorical Encoding:*
Converted text columns to pandas' category dtype and passed them through the model without using one-hot encoding which would take up heavy memory consumption (Kaggle, 2025a). However, for Logistic Regression and Random Forest, we used ‘pd.get_dummies()’ to one-hot encode the data as required by scikit-learn models.

**The validation scheme and models.**

To prevent overfitting and to ensure generalization of our model, we used the technique called “5-Fold Stratified Cross-Validation (Course lecturer,2025).”

*The baseline model* 
correctly predicts "0" (the majority class) with a baseline ROC-AUC of 0.50000.

*Logistic Regression (Linear Model)*
We applied StandardScaler to normalize the features and filled missing values with the median to prevent Convergence Warnings. OOF ROC-AUC:0.94408.

*Random Forest (Bagging Model)*
We used ‘pd.get_dummies ()’ for encoding and trained 200 trees per fold. OOF ROC-AUC:0.95692.

*LightGBM (Gradient Boosting Model)*

Our best performing model.
Results: 
  Fold 1: 0.96881
  Fold 2: 0.96785
  Fold 3: 0.96783
  Fold 4: 0.96891
  Fold 5: 0.96816
  Overall OOF ROC-AUC: 0.96831.

**Hyperparameter Tuning**

We adopted the method 'Randomizedsearchcv' for optimizing our lightGBM model. The best parameters found were ` {'num_leaves': 127, 'n_estimators': 300, 'learning_rate': 0.1} `, which gave a Cross-Validation score of 0.96784. We finally decided to use the same LightGBM model we had originally since it had slightly better validation score.

After training our three models, we selected the LightGBM model for our final Kaggle submission because it achieved the highest validation score (0.96831). We generated our final submission file using this model. 

Local OOFC ROC-AUC: 0.96831
Kaggle Public Leaderboard: 0.966918
Kaggle Private Leaderboard: 0.96905



**Team Members and Roles**

Joyce
Data loading, Exploratory Data Analysis (EDA), visualizations and writing the data dictionary.
Trustina
Feature Engineering, categorical encoding, anomaly handling, and Preprocessing. And also reviewing feature engineering code and testing XGBoost.
Rhoda
Model training (LightGBM), validation scheme (5-Fold CV), hyperparameter tuning and final ensemble and submission generation.









**References**

Course lecturer. (2025). Titanic: Exploring the data and making a first submission [Colab notebook]. [Unpublished course material].
Kaggle. (2025a). Predicting classification of bank term deposits [Notebook].
Kaggle. (2025b). Stack vs blend [Notebook].
Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31.


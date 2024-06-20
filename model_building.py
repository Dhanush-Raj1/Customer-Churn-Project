import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data\data_eda.csv')
df.head()




# feature encoding 
features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
           'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 
           'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
df_encoded = pd.get_dummies(df, columns=features, drop_first=True, dtype=int)
df_encoded.columns



                                                       
# removing CustomerID
df_encoded.drop('customerID', axis=1, inplace=True) 




# converting values of Churn
df_encoded.Churn.value_counts()
df_encoded['Churn'] = df_encoded['Churn'].replace({'No': 0, 'Yes':1})




# handling outliers 
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# visualizing to determine the presence of outliers 
# scatter plot
for col in num_cols:
    plt.scatter(df[col].value_counts().index, df[col].value_counts())
    plt.title("Distirbution of " + col)
    plt.show()

# box plot
for col in num_cols:
    plt.boxplot(df[col])
    plt.title("Distribution of " + col)
    plt.ylabel(col)
    plt.show()

def detect_outliers(df, num_cols, threshold=2.5):
    outliers = pd.DataFrame()
    for col in num_cols:
        mean = np.mean(df[col])
        std = np.std(df[col])
        z_scores = (df[col] - mean) / std
        outlier_values = df[np.abs(z_scores) > threshold]
        outliers = pd.concat([outliers, outlier_values])
    return outliers 
    
outliers_df = detect_outliers(df, num_cols)

# visualizing outliers 
plt.scatter(outliers_df['MonthlyCharges'].value_counts().index, outliers_df['MonthlyCharges'].value_counts())
plt.scatter(outliers_df['TotalCharges'].value_counts().index, outliers_df['TotalCharges'].value_counts())

# removing outliers 
df_cleaned = df_encoded.drop(outliers_df.index)

print("Size before removing outliers: ",df_encoded.shape[0])
print("Size after removing outliers: ",df_cleaned.shape[0])




# handling imbalance 
counts = df_cleaned.Churn.value_counts()
print("Percentage of not churned customers: ", round(counts.max()/counts.sum() * 100))
print("Percentage of churned customers: ", round(counts.min()/counts.sum() * 100))

# independent and dependent varaibles
X = df_cleaned.drop('Churn', axis=1)
y = df_cleaned['Churn']

# train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

## SMOTEENN 
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
print("Original X_train and y_train size: ", X_train.shape, y_train.shape)
print("Resampled X_train, y_train size: ", X_train_resampled.shape, y_train_resampled.shape)

# original distribution 
print(y_train.value_counts())
plt.bar(y_train.value_counts().index, y_train.value_counts().values, color=['lightcoral', 'brown'])
plt.title('Distribution of y_train before SMOTEENN')
plt.xlabel('y_train')
plt.ylabel('counts')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.show()

y_counts = y_train.value_counts()
minority_percentage = (y_counts.min()/y_counts.sum()) * 100
majority_percentage = (y_counts.max()/y_counts.sum()) * 100
print("The percentage of not churned customers before SMOTEENN: ", round(majority_percentage))
print("The percentage of churned customers before SMOTEENN: ", round(minority_percentage))

# resampled distribution
print(y_train_resampled.value_counts())
plt.bar(y_train_resampled.value_counts().index, y_train_resampled.value_counts().values, color=['lightcoral', 'brown'])
plt.title('Distribution of y_train after SMOTEENN')
plt.xlabel('y_train_resampled')
plt.ylabel('counts')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.show()

y_resampled_counts = y_train_resampled.value_counts()
print("The percentage of not churned customers after SMOTEENN: ", round(y_resampled_counts.min()/y_resampled_counts.sum() * 100))
print("The percentage of churned customers after SMOTEENN: ", round(y_resampled_counts.max()/y_resampled_counts.sum() * 100))




# feature scaling 
# normailization
from sklearn.preprocessing import Normalizer
normalize = Normalizer()

numerical_cols = ['MonthlyCharges', 'TotalCharges']
X_train_numerical = X_train[numerical_cols]
X_train_numerical_norm = normalize.fit_transform(X_train_numerical)

X_test_numerical = X_test[numerical_cols]
X_test_numerical_norm = normalize.transform(X_test_numerical)

X_train[numerical_cols] = X_train_numerical_norm
X_test[numerical_cols] = X_test_numerical_norm 

print(X_train, X_test)




# model building

# Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
params_lr = [{'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100, 300],
             'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cholesky']},
             
             {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10, 100, 300],
              'solver': ['liblinear', 'saga']}, 
             
             {'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10, 100, 300], 
              'solver': ['saga'], 'l1_ratio': [0.5]}]
grid_search_lr = GridSearchCV(estimator=LogisticRegression(max_iter=10000), param_grid=params_lr, cv=5, scoring='accuracy', verbose=1, n_jobs=1)
grid_search_lr.fit(X_train, y_train)

print("Best parameters for logistic regression: ", grid_search_lr.best_params_)
print("Best score for logsitc regression: ", grid_search_lr.best_score_)


# Naive bayes
from sklearn.naive_bayes import GaussianNB
params_nb = {'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-07]}
grid_search_nb = GridSearchCV(estimator=GaussianNB(), param_grid=params_nb, cv=5, scoring='accuracy', verbose=1, n_jobs=1)
grid_search_nb.fit(X_train, y_train)

print("Best parameters for naive bayes: ", grid_search_nb.best_params_)
print("Best score for naive bayes: ", grid_search_nb.best_score_)


# Knn
from sklearn.neighbors import KNeighborsClassifier
params_knn = {'n_neighbors': [3, 5, 7, 9, 11, 20, 25, 30, 40], 
              'weights': ['uniform', 'distance'],
              'algorithm': ['brute', 'kd_tree', 'ball_tree']}
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params_knn, cv=5, scoring='accuracy', verbose=1, n_jobs=1)
grid_search_knn.fit(X_train, y_train)

print("Best parameters for Knn: ", grid_search_knn.best_params_)
print("Best score for knn: ", grid_search_knn.best_score_)


# Decision tree
from sklearn.tree import DecisionTreeClassifier
params_dt = {'criterion': ['gini', 'entropy', 'log_loss'],
             'max_depth': [None, 3, 5, 10, 15, 20, 25, 30, 35, 40, 50],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4],
             'max_features': [None, 'sqrt', 'log2']}
grid_search_dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params_dt, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_dt.fit(X_train, y_train)

print("Best parameters for decision tree: ", grid_search_dt.best_params_)
print("Best score for decision tree: ", grid_search_dt.best_score_)
                              

# bagging
from sklearn.ensemble import RandomForestClassifier
params_rf = {'n_estimators': [50, 100, 150, 200],
             'criterion': ['gini', 'entropy', 'log_loss'],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4],
             'max_features': [None, 'sqrt', 'log2']}
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params_rf, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

print("Best parameters for random forest: ", grid_search_dt.best_params_)
print("Best score for random forest: ", grid_search_rf.best_score_)


# boosting
# adaboost
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.svm import SVC

dt = DecisionTreeClassifier()
lr = LogisticRegression(max_iter=10000)
svc = SVC(probability=True)

estimators = [dt, lr, svc]

params_ab = {'estimator': [dt, lr, svc],
             'n_estimators': [10, 50, 100, 200, 300, 400],
             'learning_rate': [0.01, 0.1, 0.5, 1.0]}
grid_search_ab = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=params_ab, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_ab.fit(X_train, y_train)

print("Best parameters for Adaboost: ", grid_search_ab.best_params_)
print("Best score for Adaboost : ", grid_search_ab.best_score_)


# gradient boost
from sklearn.ensemble import GradientBoostingClassifier
params_gb = {'n_estimators': [50, 100, 150, 200, 300, 500],
             'learning_rate': [0.01, 0.1, 0.2, 0.3],
             'max_depth': [3, 5, 7, 9, 11, 15],
             'subsample': [0.6, 0.8, 1.0],
             'max_features': [None, 'sqrt', 'log2']}
grid_search_gb = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=params_gb, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_gb.fit(X_train, y_train)

print("Best parameters for gradient boosting: ", grid_search_gb.best_params_)
print("Best score for gradient boosting: ", grid_search_gb.best_score_)


# xgboost
from xgboost import XGBClassifier
params_xg = {'n_estimator': [50, 100, 200],
             'learning_rate': [0.01, 0.1, 0.2, 0.3],
             'max_depth': [3, 5, 7, 10],
             'subsample': [0.6, 0.8, 1.0],
             'colsample_bytree': [0.6, 0.8, 1.0]}
grid_search_xg = GridSearchCV(estimator=XGBClassifier(), param_grid=params_xg, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_xg.fit(X_train, y_train)

print("Best parameters for xg boost: ", grid_search_xg.best_params_)
print("Best score for xg boost: ", grid_search_xg.best_score_)

 
# SVM
from sklearn.svm import SVC
params_svm = {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'gamma': ['scale', 'auto']}
grid_search_svm = GridSearchCV(estimator=SVC(), param_grid=params_svm, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)

print("Best parameter for SVC: ", grid_search_svm.best_params_)
print("Best score for SVC: ", grid_search_svm.best_score_)
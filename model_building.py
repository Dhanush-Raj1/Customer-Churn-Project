import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv('data\data_cleaned.csv')
df.head()

# grouping tenure in to categories of 12 months 
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
df['tenure'] = pd.cut(df['tenure'], range(1, 80, 12), right=False, labels=labels)




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
for col in num_cols:
    plt.scatter(df[col].value_counts().index, df[col].value_counts())
    plt.title("Distirbution of " + col)
    plt.show()





# handling imbalance 
counts = df_encoded.Churn.value_counts()
print("Percentage of not churned customers: ", round(counts.max()/counts.sum() * 100))
print("Percentage of churned customers: ", round(counts.min()/counts.sum() * 100))

# independent and dependent varaibles
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

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

y_res_counts = y_train_resampled.value_counts()
print("The percentage of not churned customers after SMOTEENN: ", round(y_res_counts.min()/y_res_counts.sum() * 100))
print("The percentage of churned customers after SMOTEENN: ", round(y_res_counts.max()/y_res_counts.sum() * 100))




# feature scaling 



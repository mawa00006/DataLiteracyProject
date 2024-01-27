import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt
from src.data_preprocessing import read_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score)
from sklearn.metrics import f1_score

# Read the data
data = pd.read_csv("../dat/preprocessed_brca_metabric_clinical_data.tsv", sep='\t')


# Type of Breast Surgery -----------------------------------------------------------------------------------

# preprocessing ---------------------------------------------------------------
# exclude the NaNs
data = data.dropna(subset=['Type of Breast Surgery', 'Tumor Size', 'Tumor Stage',
                                 'Neoplasm Histologic Grade', 'Lymph nodes examined positive',
                                 'Mutation Count', 'Integrative Cluster'])
print(data.shape)  # (1289, 33)

# convert categorical to dummy numerical variable
tumor_type_mapping = {'Breast Invasive Ductal Carcinoma': 0,
                      'Breast Mixed Ductal and Lobular Carcinoma': 1,
                      'Breast Invasive Lobular Carcinoma': 2,
                      'Invasive Breast Carcinoma': 3}
cluster_mapping = {'1': 0, '2': 1, '3': 2, '4ER+': 3, '4ER-': 4, '5': 5,
                   '6': 6, '7': 7, '8': 8, '9': 9, '10': 10}

data['Cancer Type Detailed'] = data['Cancer Type Detailed'].map(tumor_type_mapping)
data['Integrative Cluster'] = data['Integrative Cluster'].map(cluster_mapping)

# histogram of all compared to those that received Mastectomy -----------------
mastectomy = data[data['Type of Breast Surgery'] == 1]
conserving = data[data['Type of Breast Surgery'] == 0]

plt.hist(data['Age at Diagnosis'], bins=20, alpha=0.5, label='all')
plt.hist(mastectomy['Age at Diagnosis'], bins=20, alpha=0.5, label='mastectomy')
plt.xlabel('Age at Diagnosis [years]')
plt.ylabel('Number of Patients')
plt.legend()
plt.show()

# simple logistic regression - with Age at Diagnosis --------------------------

# splitting into training and test set and
# defining the dependent and independent variables
X_train, X_test, y_train, y_test = train_test_split(data[['Age at Diagnosis']],
                                                    data[['Type of Breast Surgery']], test_size=0.25, random_state=42)
# print(X_train.shape)  # (966, 1)
# print(y_train.shape)  # (966, 1)
# print(X_test.shape)  # (323, 1)
# print(y_test.shape)  # (323, 1)

# building the model and fitting the data
log_reg = sm.Logit(y_train, X_train).fit()
print(log_reg.summary())

# performing predictions on the test dataset
pred = log_reg.predict(X_test)
prediction = list(map(round, pred))

# comparing original and predicted values of y
print('Actual values', list(y_test.values))
print('Predictions :', prediction)

# confusion matrix
cm = confusion_matrix(y_test, prediction)
print("Confusion Matrix : \n", cm)

# accuracy score
print('Test accuracy = ', accuracy_score(y_test, prediction))

# F1-score
f_score = f1_score(y_test, prediction)
print("F1 score : \n", f_score)


# multiple logistic regression - including more variables ---------------------

features = ['Age at Diagnosis', 'ER Status', 'Neoplasm Histologic Grade',
                'HER2 Status', 'Lymph nodes examined positive', 'Mutation Count',
                'PR Status', 'Tumor Size', 'Tumor Stage', 'Cancer Type Detailed',
                'Integrative Cluster']

# splitting into training and test set and
# defining the dependent and independent variables
X_train, X_test, y_train, y_test = train_test_split(
    data[features],
    data[['Type of Breast Surgery']], test_size=0.25, random_state=42)

# building the model and fitting the data
log_reg = sm.Logit(y_train, X_train).fit()
print(log_reg.summary())

# correcting for multiple testing
reject, adjusted_p_values, _, _ = multipletests(log_reg.pvalues, method='bonferroni')
df = pd.DataFrame({'variable': features,
                    'p_adjusted': adjusted_p_values,
                    'significant': reject})
print(df)

# performing predictions on the test dataset
pred = log_reg.predict(X_test)
prediction = list(map(round, pred))

# comparing original and predicted values of y
print('Actual values', list(y_test.values))
print('Predictions :', prediction)

# confusion matrix
cm = confusion_matrix(y_test, prediction)
print("Confusion Matrix : \n", cm)

# accuracy score
print('Test accuracy = ', accuracy_score(y_test, prediction))

# F1-score
f_score = f1_score(y_test, prediction)
print("F1 score : \n", f_score)

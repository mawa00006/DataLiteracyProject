import numpy as np
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from src.data_preprocessing import read_and_preprocess


# Read the data
data = pd.read_csv("../dat/preprocessed_brca_metabric_clinical_data.tsv", sep='\t')

### Age vs Therapy ------------------------------

# boxplots width according to sample size - sqrt(n)
# Hormone Therapy - ['YES' 'NO' nan]
# Chemotherapy - ['NO' 'YES' nan]
# Type of Breast Surgery - ['MASTECTOMY' 'BREAST CONSERVING' nan]
# Radio Therapy - ['YES' 'NO' nan]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sample_size_sqrt_ht = np.sqrt(data.groupby('Hormone Therapy').size()) / 50
sample_size_sqrt_ct = np.sqrt(data.groupby('Chemotherapy').size()) / 50
sample_size_sqrt_bs = np.sqrt(data.groupby('Type of Breast Surgery').size()) / 50
sample_size_sqrt_rt = np.sqrt(data.groupby('Radio Therapy').size()) / 50

# Boxplot for Hormone Therapy
sns.boxplot(x='Hormone Therapy', y='Age at Diagnosis', data=data, ax=axes[0, 0], width=sample_size_sqrt_ht)
axes[0, 0].set_title('Age at Diagnosis by Hormone Therapy')
# Boxplot for Chemotherapy
sns.boxplot(x='Chemotherapy', y='Age at Diagnosis', data=data, ax=axes[0, 1], width=sample_size_sqrt_ct)
axes[0, 1].set_title('Age at Diagnosis by Chemotherapy')
# Boxplot for Type of Breast Surgery
sns.boxplot(x='Type of Breast Surgery', y='Age at Diagnosis', data=data, ax=axes[1, 0], width=sample_size_sqrt_bs)
axes[1, 0].set_title('Age at Diagnosis by Type of Breast Surgery')
axes[1, 0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better visibility
# Boxplot for Radio Therapy
sns.boxplot(x='Radio Therapy', y='Age at Diagnosis', data=data, ax=axes[1, 1], width=sample_size_sqrt_rt)
axes[1, 1].set_title('Age at Diagnosis by Radio Therapy')

plt.tight_layout()
plt.show()


# Chemotherapy ------------------------------------------------------------------------------------------
data_chemo = data.dropna(subset=['Chemotherapy', 'Tumor Size', 'Tumor Stage'])
print(data_chemo.shape) # (1735, 33)
chemo = data_chemo[data_chemo['Chemotherapy'] == 1]
no_chemo = data_chemo[data_chemo['Chemotherapy'] == 0]

# histogram of all compared to those that received chemotherapy
plt.hist(data_chemo['Age at Diagnosis'], bins=20, alpha=0.5, label='all')
plt.hist(chemo['Age at Diagnosis'], bins=20, alpha=0.5, label='chemotherapy')
plt.xlabel('Age at Diagnosis [years]')
plt.ylabel('Number of Patients')
plt.legend()
plt.show()

# logistic regression - with Age at Diagnosis

# defining the dependent and independent variables
Xtrain = data_chemo[['Age at Diagnosis']]
ytrain = data_chemo[['Chemotherapy']]

# building the model and fitting the data
log_reg = sm.Logit(ytrain, Xtrain).fit()
print('Chemotherapy - Age at Diagnosis -------------------------------------------------------')
print(log_reg.summary())

# Plotting the logistic regression curve
# xxx fixme


# logistic regression - including more variables

# defining the dependent and independent variables
Xtrain = data_chemo[['Age at Diagnosis', 'Tumor Size', 'Tumor Stage']]
ytrain = data_chemo[['Chemotherapy']]

# building the model and fitting the data
log_reg = sm.Logit(ytrain, Xtrain).fit()
print('Chemotherapy - Age at Diagnosis, Tumor Size, Tumor Stage ------------------------------')
print(log_reg.summary())


# Type of Breast Surgery ------------------------------------------------------------------------------------------
data_surgery = data.dropna(subset=['Type of Breast Surgery', 'Tumor Size', 'Tumor Stage'])
print(data_surgery.shape) # (1735, 33)
surgery = data_surgery[data_surgery['Type of Breast Surgery'] == 1]
no_surgery = data_surgery[data_surgery['Type of Breast Surgery'] == 0]

# histogram of all compared to those that received chemotherapy
plt.hist(data_surgery['Age at Diagnosis'], bins=20, alpha=0.5, label='all')
plt.hist(surgery['Age at Diagnosis'], bins=20, alpha=0.5, label='Mastectomy')
plt.xlabel('Age at Diagnosis [years]')
plt.ylabel('Number of Patients')
plt.legend()
plt.show()

# logistic regression - with Age at Diagnosis

# defining the dependent and independent variables
Xtrain = data_surgery[['Age at Diagnosis']]
ytrain = data_surgery[['Type of Breast Surgery']]

# building the model and fitting the data
log_reg = sm.Logit(ytrain, Xtrain).fit()
print('Type of Breast Surgery - Age at Diagnosis -------------------------------------------------------')
print(log_reg.summary())

# logistic regression - including more variables

# defining the dependent and independent variables
Xtrain = data_surgery[['Age at Diagnosis', 'Tumor Size', 'Tumor Stage']]
ytrain = data_surgery[['Type of Breast Surgery']]


# Hormone Therapy ------------------------------------------------------------------------------------------

# ER+ Status!!!!
data_hormone = data.dropna(subset=['Hormone Therapy', 'Tumor Size', 'Tumor Stage', 'ER Status'])
print(data_hormone.shape) # (1735, 33)
hormone = data_hormone[data_hormone['Hormone Therapy'] == 1]
no_hormone = data_hormone[data_hormone['Hormone Therapy'] == 0]

# histogram of all compared to those that received chemotherapy
plt.hist(data_hormone['Age at Diagnosis'], bins=20, alpha=0.5, label='all')
plt.hist(hormone['Age at Diagnosis'], bins=20, alpha=0.5, label='Hormone Therapy')
plt.xlabel('Age at Diagnosis [years]')
plt.ylabel('Number of Patients')
plt.legend()
plt.show()

# logistic regression - with Age at Diagnosis

# defining the dependent and independent variables
Xtrain = data_hormone[['Age at Diagnosis']]
ytrain = data_hormone[['Hormone Therapy']]

# building the model and fitting the data
log_reg = sm.Logit(ytrain, Xtrain).fit()
print('Hormone Therapy - Age at Diagnosis -------------------------------------------------------')
print(log_reg.summary())

# logistic regression - including more variables

# defining the dependent and independent variables
Xtrain = data_hormone[['Age at Diagnosis', 'Tumor Size', 'Tumor Stage', 'ER Status']]
ytrain = data_hormone[['Hormone Therapy']]

# building the model and fitting the data
log_reg = sm.Logit(ytrain, Xtrain).fit()
print('Hormone Therapy - Age at Diagnosis, Tumor Size, Tumor Stage ------------------------------')
print(log_reg.summary())


# Radio Therapy ------------------------------------------------------------------------------------------
data_radio = data.dropna(subset=['Radio Therapy', 'Tumor Size', 'Tumor Stage'])
print(data_radio.shape) # (1735, 33)
radio = data_radio[data_hormone['Radio Therapy'] == 1]
no_radio = data_radio[data_hormone['Radio Therapy'] == 0]

# histogram of all compared to those that received chemotherapy
plt.hist(data_radio['Age at Diagnosis'], bins=20, alpha=0.5, label='all')
plt.hist(radio['Age at Diagnosis'], bins=20, alpha=0.5, label='Radio Therapy')
plt.xlabel('Age at Diagnosis [years]')
plt.ylabel('Number of Patients')
plt.legend()
plt.show()

# logistic regression - with Age at Diagnosis

# defining the dependent and independent variables
Xtrain = data_radio[['Age at Diagnosis']]
ytrain = data_radio[['Radio Therapy']]

# building the model and fitting the data
log_reg = sm.Logit(ytrain, Xtrain).fit()
print('Radio Therapy - Age at Diagnosis -------------------------------------------------------')
print(log_reg.summary())

# logistic regression - including more variables

# defining the dependent and independent variables
Xtrain = data_radio[['Age at Diagnosis', 'Tumor Size', 'Tumor Stage']]
ytrain = data_radio[['Radio Therapy']]

# building the model and fitting the data
log_reg = sm.Logit(ytrain, Xtrain).fit()
print('Radio Therapy - Age at Diagnosis, Tumor Size, Tumor Stage ------------------------------')
print(log_reg.summary())





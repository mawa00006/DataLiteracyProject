from data_reader import read_and_preprocess
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lifelines import CoxPHFitter


#fetch data frame
df=read_and_preprocess('brca_metabric_clinical_data.tsv')
df = df.dropna(subset=['Age at Diagnosis', 'Overall Survival (Months)', 'Overall Survival Status', 'Chemotherapy', 'Nottingham prognostic index','Relapse Free Status (Months)', 'Relapse Free Status','Radio Therapy'])
#filter for age
#df = df[df['Age at Diagnosis'] >= 70]
#filter for cancer type
#df = df[df['Cancer Type Detailed'] == 'Breast Invasive Lobular Carcinoma']

# Filter the dataset for patients who received chemotherapy
chemo_patients = df[df['Chemotherapy'] == 1]

# Filter the dataset for patients who did not receive chemotherapy
no_chemo_patients = df[df['Chemotherapy'] == 0]

'''# Kaplan-Meier estimator for patients who received chemotherapy
kmf_chemo = KaplanMeierFitter()
kmf_chemo.fit(durations=chemo_patients['Overall Survival (Months)'], event_observed=chemo_patients['Overall Survival Status']=="1:DECEASED" , label='Chemotherapy')

# Kaplan-Meier estimator for patients who did not receive chemotherapy
kmf_no_chemo = KaplanMeierFitter()
kmf_no_chemo.fit(durations=no_chemo_patients['Overall Survival (Months)'], event_observed=no_chemo_patients['Overall Survival Status']=="1:DECEASED", label='No_Chemotherapy')

# Plot the Kaplan-Meier curves
plt.figure(figsize=(10, 6))
kmf_chemo.plot_survival_function()
kmf_no_chemo.plot_survival_function()
plt.title('Kaplan-Meier Curves for Overall Survival')
plt.xlabel('Time (Months)')
plt.ylabel('Survival Probability')
plt.show()'''


# creating binary survival status 
df['Overall Survival Status'] = df['Overall Survival Status'].map({'0:LIVING': 0, '1:DECEASED': 1})
df['Relapse Free Status'] = df['Relapse Free Status'].map({'0:Not Recurred': 0, '1:Recurred': 1})

# Filter the dataset for patients with a score greater than 4 in Nottingham prognostic index
#df =  df[(df['Nottingham prognostic index'] > 4) & (df['Nottingham prognostic index'] < 5.5)]
# Filter the dataset for wanted features 
'''cox_data = df[['Age at Diagnosis', 'Nottingham prognostic index', 'Chemotherapy', 'Overall Survival (Months)', 'Overall Survival Status','Radio Therapy']]

# Fit Cox Proportional Hazard model
cph = CoxPHFitter()
cph.fit(cox_data, duration_col='Overall Survival (Months)', event_col='Overall Survival Status')

# Display summary of the model
print(cph.summary)

# Plot the coefficients
cph.plot()
plt.title('Cox Proportional Hazard Model Coefficients')
plt.show()'''

#cox for replapse
cox_data = df[['Age at Diagnosis', 'Nottingham prognostic index', 'Chemotherapy', 'Relapse Free Status (Months)', 'Relapse Free Status','Radio Therapy']]

# Fit Cox Proportional Hazard model
cph = CoxPHFitter()
cph.fit(cox_data, duration_col='Relapse Free Status (Months)', event_col='Relapse Free Status')

# Display summary of the model
print(cph.summary)

# Plot the coefficients
cph.plot()
plt.title('Cox Proportional Hazard Model Coefficients')
plt.show()

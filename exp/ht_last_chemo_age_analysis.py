import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from lifelines.statistics import logrank_test

# Read the TSV file into a DataFrame
df = pd.read_csv('preprocessed_brca_metabric_clinical_data.tsv', sep='\t')
df = df.dropna(subset=['Age at Diagnosis', 'Overall Survival (Months)', 'Overall Survival Status', 'Chemotherapy', 'Relapse Free Status (Months)', 'Relapse Free Status', 'Tumor Stage'])

# creating binary survival status 
df['Overall Survival Status'] = df['Overall Survival Status'].map({'0:LIVING': 0, '1:DECEASED': 1})
df['Relapse Free Status'] = df['Relapse Free Status'].map({'0:Not Recurred': 0, '1:Recurred': 1})
df['PR Status'] = df['PR Status'].map({'Positive': 1, 'Negative': 0})
df['ER Status'] = df['ER Status'].map({'Positive': 1, 'Negative': 0})


#creat plots for tumor stages (1,1),(2,2),(3,3)
#grouping patients
group1 = df[
    
    (df['Chemotherapy Binary'] == 1) &
    (df['Age at Diagnosis'] >= 50) &
    (df['Tumor Stage'] == 1)
]

group2 = df[
    (df['Chemotherapy Binary'] == 1) &
    (df['Age at Diagnosis'] < 50) &
    (df['Tumor Stage'] == 1)
]
# creat plots for tumor stages (1,1),(2,2),(3,3)
# Create Kaplan-Meier estimators for relapse for each group
kmf_group1 = KaplanMeierFitter()
kmf_group2 = KaplanMeierFitter()

# Fit the Kaplan-Meier curves for each group
kmf_group1.fit(durations=group1['Relapse Free Status (Months)'], event_observed=group1['Relapse Free Status'], label='Over 50')
kmf_group2.fit(durations=group2['Relapse Free Status (Months)'], event_observed=group2['Relapse Free Status'], label='Under 50')

# Plot the Kaplan-Meier curves for overall survival
plt.figure(figsize=(10, 6))
kmf_group1.plot(color='blue')
kmf_group2.plot(color='orange')

# Customize the plot
plt.title('Kaplan-Meier Curve - Relapse Survival')
plt.xlabel('Time (Months)')
plt.ylabel('Relapse Survival Probability')
plt.legend()
plt.show()


#
# Create Kaplan-Meier estimator for death survival each group
kmf_group3 = KaplanMeierFitter()
kmf_group4 = KaplanMeierFitter()

# Fit the Kaplan-Meier curves for each group
kmf_group3.fit(durations=group1['Overall Survival (Months)'], event_observed=group1['Overall Survival Status'], label='Over 50')
kmf_group4.fit(durations=group2['Overall Survival (Months)'], event_observed=group2['Overall Survival Status'], label='Under 50')

# Plot the Kaplan-Meier curves
plt.figure(figsize=(10, 6))
kmf_group3.plot(color='blue')
kmf_group4.plot(color='orange')

# Customize the plot
plt.title('Kaplan-Meier Curve - Overall Survival')
plt.xlabel('Time (Months)')
plt.ylabel('Death Survival Probability')
plt.legend()
plt.show()

# creat table 3x3 for each pair (1,1),(2,2),(3,3)
# Perform log-rank test for relapse
results_relapse = logrank_test(group1['Relapse Free Status (Months)'], group2['Relapse Free Status (Months)'], 
                       event_observed_A=group1['Relapse Free Status'], event_observed_B=group2['Relapse Free Status'])

# Display the test statistic and p-value
print(f'Log-Rank Test Statistic: {results_relapse.test_statistic:.3f}')
print(f'P-value: {results_relapse.p_value:.3f}')

# Interpret the results
alpha = 0.05
if results_relapse.p_value < alpha:
    print('The relapse survival curves are significantly different. Reject the null hypothesis.')
else:
    print('There is no significant difference in relaapse survival curves. Fail to reject the null hypothesis.')
# creat table 3x3 for each pair (1,1),(2,2),(3,3)
# Perform log-rank test for survival from death
results_surv = logrank_test(group1['Overall Survival (Months)'], group2['Overall Survival (Months)'], 
                       event_observed_A=group1['Overall Survival Status'], event_observed_B=group2['Overall Survival Status'])

# Display the test statistic and p-value
print(f'Log-Rank Test Statistic: {results_surv.test_statistic:.3f}')
print(f'P-value: {results_surv.p_value:.3f}')
# Interpret the results
alpha = 0.05
if results_surv.p_value < alpha:
    print('The survival curves are significantly different. Reject the null hypothesis.')
else:
    print('There is no significant difference in survival curves. Fail to reject the null hypothesis.')


#fetch CI from kaplan mayer model
#confidence_interval_group1 = kmf_group1.confidence_interval_ 
#confidence_interval_group2 =  kmf_group2.confidence_interval_
#confidence_interval_group3 =  kmf_group3.confidence_interval_
#confidence_interval_group4 =  kmf_group3.confidence_interval_



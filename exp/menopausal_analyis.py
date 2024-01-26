import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
import seaborn as sns
import numpy as np
import statsmodels.api as sm

# Read the TSV file into a DataFrame
df = pd.read_csv('brca_metabric_clinical_data.tsv', sep='\t')

# Display the first few rows of the DataFrame

filtered_df = df.dropna(subset=['Inferred Menopausal State', "Patient's Vital Status", "Relapse Free Status"])
filtered_df = filtered_df[filtered_df["Patient's Vital Status"] != 'Died of Other Causes']
filtered_df['Relapse_Status'] = filtered_df['Relapse Free Status'].apply(lambda x: 1 if x == '1:Recurred' else (0 if x == '0:Not Recurred' else None))
filtered_df['binary_menopausal_status'] = filtered_df['Inferred Menopausal State'].apply(lambda x: 1 if 'Post' in str(x) else (0 if 'Pre' in str(x) else None))
#get the er postive rows
#filtered_df = filtered_df[filtered_df['HER2 Status'] == 'Negative']
#get the 
#filtered_df = filtered_df[filtered_df['HER2 Status'] == 'Positive']

# Assuming df is your DataFrame
'''kmf = KaplanMeierFitter()

# Separating data into menopausal and non-menopausal groups
menopausal = filtered_df['Inferred Menopausal State'] == 'Post'
non_menopausal = filtered_df['Inferred Menopausal State'] == 'Pre'



# Fitting the Kaplan-Meier curve for the menopausal group
kmf.fit(filtered_df['Relapse Free Status (Months)'][menopausal], event_observed=filtered_df['Relapse_Status'][menopausal], label='Post menopausal')
ax = kmf.plot()

# Fitting the Kaplan-Meier curve for the non-menopausal group
kmf.fit(filtered_df['Relapse Free Status (Months)'][non_menopausal], event_observed=filtered_df['Relapse_Status'][non_menopausal], label='Pre menopausal')

kmf.plot(ax=ax)

# Plotting settings
#plt.xlabel('Time in months')
#plt.ylabel(' relapse free Survival Probability')
#plt.title('Kaplan-Meier Curves by Menopausal State for all cancer types')
#plt.show()

time_data = filtered_df['Relapse Free Status (Months)']
event_data = filtered_df['Relapse_Status']
results = logrank_test(time_data[menopausal], time_data[non_menopausal], event_observed_A=event_data[menopausal], event_observed_B=event_data[non_menopausal])
p_value = results.p_value
print(f"P-value: {p_value}")

time_data_menopausal = filtered_df['Relapse Free Status (Months)'][menopausal]
time_data_non_menopausal = filtered_df['Relapse Free Status (Months)'][non_menopausal]
event_data_menopausal = filtered_df['Relapse_Status'][menopausal]
event_data_non_menopausal = filtered_df['Relapse_Status'][non_menopausal]

# Perform manual bootstrapping by resampling and running log-rank test
n_iterations = 1000
p_values = []
for _ in range(n_iterations):
    # Sample with replacement for both groups
    bootstrapped_menopausal = np.random.choice(time_data_menopausal, size=len(time_data_menopausal), replace=True)
    bootstrapped_non_menopausal = np.random.choice(time_data_non_menopausal, size=len(time_data_non_menopausal), replace=True)
    
    # Run log-rank test on bootstrapped samples
    results = logrank_test(bootstrapped_menopausal, bootstrapped_non_menopausal, event_observed_A=event_data_menopausal, event_observed_B=event_data_non_menopausal)
    p_values.append(results.p_value)

# Calculate and print the average p-value from bootstrapping
average_p_value = np.mean(p_values)
print(f"Average Bootstrapped Log-Rank Test p-value: {average_p_value}")'''

'''from scipy.stats import chi2_contingency

# Contingency table between 'Inferred Menopausal State' and 'Cellularity'
contingency_table = pd.crosstab(filtered_df['Inferred Menopausal State'], filtered_df['Cellularity'])

# Performing chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"P-value: {p}")'''


'''filtered_df = filtered_df[filtered_df['Cancer Type Detailed'] == 'Metaplastic Breast Cancer']
# Assuming 'data' is your DataFrame containing 'Age at Diagnosis' and 'Cancer Type'

# Create age groups
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Define age bins
labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']  # Labels for bins
filtered_df['Age Group'] = pd.cut(filtered_df['Age at Diagnosis'], bins=bins, labels=labels)

# Group data by Age Group and Cancer Type and count occurrences
grouped = filtered_df.groupby(['Age Group', 'Cancer Type Detailed']).size().unstack()

# Plotting
grouped.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.title('Frequency of Cancer Types by Age Group')
plt.legend(title='Cancer Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''
# Assuming 'data' is your DataFrame containing 'Age at Diagnosis', 'PR Status', 'ER Status', and 'HER2 Status'
'''filtered_df = filtered_df[filtered_df['Inferred Menopausal State'] == 'Post']
# Filter data for positive PR, ER, and HER2 statuses separately
pr_positive = filtered_df[filtered_df['PR Status'] == 'Positive']
er_positive = filtered_df[filtered_df['ER Status'] == 'Positive']
her2_positive = filtered_df[filtered_df['HER2 Status'] == 'Positive']

# Define age groups
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Define age bins
labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']  # Labels for bins

# Create 'Age Group' column based on 'Age at Diagnosis'
pr_positive['Age Group'] = pd.cut(pr_positive['Age at Diagnosis'], bins=bins, labels=labels)
er_positive['Age Group'] = pd.cut(er_positive['Age at Diagnosis'], bins=bins, labels=labels)
her2_positive['Age Group'] = pd.cut(her2_positive['Age at Diagnosis'], bins=bins, labels=labels)

# Count occurrences in each age group for each positive status
pr_counts = pr_positive['Age Group'].value_counts().sort_index()
er_counts = er_positive['Age Group'].value_counts().sort_index()
her2_counts = her2_positive['Age Group'].value_counts().sort_index()

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

pr_counts.plot(kind='bar', ax=axes[0], color='red')
axes[0].set_title('PR Positive Frequency by Age Group')
axes[0].set_xlabel('Age Group')
axes[0].set_ylabel('Frequency')
axes[0].tick_params(axis='x', rotation=45)

er_counts.plot(kind='bar', ax=axes[1], color='blue')
axes[1].set_title('ER Positive Frequency by Age Group')
axes[1].set_xlabel('Age Group')
axes[1].set_ylabel('Frequency')
axes[1].tick_params(axis='x', rotation=45)

her2_counts.plot(kind='bar', ax=axes[2], color='green')
axes[2].set_title('HER2 Positive Frequency by Age Group')
axes[2].set_xlabel('Age Group')
axes[2].set_ylabel('Frequency')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()'''

# Probability of having HER2 positive, ER positive, and PR positive given 'Inferred Menopausal State' is 'Pre'
'''pre_menopausal = filtered_df[filtered_df['Inferred Menopausal State'] == 'Pre']

# Probability of having HER2 positive given 'Inferred Menopausal State' is 'Pre'
prob_her2_pre = pre_menopausal[pre_menopausal['HER2 Status'] == 'Positive'].shape[0] / pre_menopausal.shape[0]

# Probability of having ER positive given 'Inferred Menopausal State' is 'Pre'
prob_er_pre = pre_menopausal[pre_menopausal['ER Status'] == 'Positive'].shape[0] / pre_menopausal.shape[0]

# Probability of having PR positive given 'Inferred Menopausal State' is 'Pre'
prob_pr_pre = pre_menopausal[pre_menopausal['PR Status'] == 'Positive'].shape[0] / pre_menopausal.shape[0]

# Probability of having HER2 positive, ER positive, and PR positive given 'Inferred Menopausal State' is 'Post'
post_menopausal = filtered_df[filtered_df['Inferred Menopausal State'] == 'Post']

# Probability of having HER2 positive given 'Inferred Menopausal State' is 'Post'
prob_her2_post = post_menopausal[post_menopausal['HER2 Status'] == 'Positive'].shape[0] / post_menopausal.shape[0]

# Probability of having ER positive given 'Inferred Menopausal State' is 'Post'
prob_er_post = post_menopausal[post_menopausal['ER Status'] == 'Positive'].shape[0] / post_menopausal.shape[0]

# Probability of having PR positive given 'Inferred Menopausal State' is 'Post'
prob_pr_post = post_menopausal[post_menopausal['PR Status'] == 'Positive'].shape[0] / post_menopausal.shape[0]

print("Probabilities for Pre-Menopausal Patients:")
print(f"HER2 Positive Probability: {prob_her2_pre:.2f}")
print(f"ER Positive Probability: {prob_er_pre:.2f}")
print(f"PR Positive Probability: {prob_pr_pre:.2f}")

print("\nProbabilities for Post-Menopausal Patients:")
print(f"HER2 Positive Probability: {prob_her2_post:.2f}")
print(f"ER Positive Probability: {prob_er_post:.2f}")
print(f"PR Positive Probability: {prob_pr_post:.2f}")'''
'''clean_df = filtered_df.dropna()
X = clean_df[[ 'Tumor Size', 'Age at Diagnosis']]  # Independent variables
y = clean_df['Relapse Free Status (Months)']  # Dependent variable

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the ordinary least squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Display summary of the regression model
print(model.summary())'''
from lifelines import CoxPHFitter

# Replace spaces and special characters in column names with underscores
filtered_df.columns = filtered_df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
# Check lengths of columns
print(filtered_df[['TMB_nonsynonymous', 'Tumor_Size', 'Tumor_Stage']].count())

# Handle missing values if present
filtered_df.dropna(subset=['TMB_nonsynonymous', 'Tumor_Size', 'Tumor_Stage', "Age_at_Diagnosis","Nottingham_prognostic_index"], inplace=True)


# Create a Cox proportional hazards model
cph = CoxPHFitter()

# Fit the model using modified column names
cph.fit(filtered_df, duration_col='Relapse_Free_Status_Months', event_col='Relapse_Status', formula='TMB_nonsynonymous + Tumor_Size + Tumor_Stage + Age_at_Diagnosis + Nottingham_prognostic_index ')

# Display summary of the fitted model
cph.print_summary()






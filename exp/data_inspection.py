import pylab
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from src.data_preprocessing import read_and_preprocess
from scipy.stats import f_oneway
from scipy.stats import ttest_ind

# Read the data
data = pd.read_csv("../dat/preprocessed_brca_metabric_clinical_data.tsv", sep='\t')

### inspect the data ------------------------------

# For Overall Survival Status '0:LIVING' what is the Overall Survival (Months)
data_living = data[data['Overall Survival Status'] == '0:LIVING']
plt.hist(data_living['Overall Survival (Months)'],bins=20)
plt.xlabel('Overall Survival (Months)')
plt.ylabel('Number of Patients')
plt.show()
#print(data_living['Overall Survival (Months)'].max()) # 337.0333333

# Age at Diagnosis (mean = 60.355; min = 21.93; max = 96.29)

# normal distribution for ...
# ...Age at Diagnosis
# histogram
plt.hist(data['Age at Diagnosis'], bins=20)
plt.xlabel('Age at Diagnosis [years]')
plt.ylabel('Counts')
plt.show()
# qq plot
stats.probplot(data['Age at Diagnosis'], dist="norm", plot=pylab)
pylab.show()

# ...Tumor Size
# histogram
plt.hist(data['Tumor Size'], bins=20)
plt.xlabel('Tumor Size')
plt.ylabel('Counts')
plt.show()
# qq plot
stats.probplot(data['Tumor Size'], dist="norm", plot=pylab)
pylab.show()

# ...Mutation Count
# histogram
plt.hist(data['Mutation Count'], bins=20)
plt.xlabel('Mutation Count')
plt.ylabel('Counts')
plt.show()
# qq plot
stats.probplot(data['Mutation Count'], dist="norm", plot=pylab)
pylab.show()

# ...Lymph nodes examined positive
# histogram
plt.hist(data['Lymph nodes examined positive'], bins=20)
plt.xlabel('Lymph nodes examined positive')
plt.ylabel('Counts')
plt.show()
# qq plot
stats.probplot(data['Lymph nodes examined positive'], dist="norm", plot=pylab)
pylab.show()


# Age at Diagnosis compared to other variables  ------------------------------
fig, axes = plt.subplots(4, 4, figsize=(12, 10))

sns.scatterplot(x='Tumor Size', y='Age at Diagnosis', data=data, ax=axes[0, 0])
sns.scatterplot(x='Mutation Count', y='Age at Diagnosis', data=data, ax=axes[0, 1])
sns.scatterplot(x='Lymph nodes examined positive', y='Age at Diagnosis', data=data, ax=axes[0, 2])

sns.boxplot(x='Tumor Stage', y='Age at Diagnosis', data=data, ax=axes[1, 0])
sns.boxplot(x='Cellularity', y='Age at Diagnosis', data=data, ax=axes[1, 1])
sns.boxplot(x='ER Status', y='Age at Diagnosis', data=data, ax=axes[1, 2])
sns.boxplot(x='Inferred Menopausal State', y='Age at Diagnosis', data=data, ax=axes[1, 3])

sns.boxplot(x='Overall Survival Status', y='Age at Diagnosis', data=data, ax=axes[2, 0])
sns.boxplot(x='Relapse Free Status', y='Age at Diagnosis', data=data, ax=axes[2, 1])

sns.boxplot(x='Type of Breast Surgery', y='Age at Diagnosis', data=data, ax=axes[3, 0])
sns.boxplot(x='Chemotherapy', y='Age at Diagnosis', data=data, ax=axes[3, 1])
sns.boxplot(x='Hormone Therapy', y='Age at Diagnosis', data=data, ax=axes[3, 2])
sns.boxplot(x='Radio Therapy', y='Age at Diagnosis', data=data, ax=axes[3, 3])

plt.tight_layout()
plt.show()


# correlation coefficient between Age at Diagnosis and...
# (spearman for non-linear relationships better)
# ...Tumor Size
data_ts = data.dropna(subset=['Age at Diagnosis', 'Tumor Size'])  # exclude NAs
result_pearson = pearsonr(data_ts['Age at Diagnosis'], data_ts['Tumor Size'])
result_spearman = spearmanr(data_ts['Age at Diagnosis'], data_ts['Tumor Size'])
print('Age at Diagnosis and Tumor Size')
print('Pearson Correlation: ' + str(result_pearson.pvalue) + '\n'
      + 'Spearman Correlation: ' + str(result_spearman.pvalue))
print('---------------------------------------------------------')
# both are significant ()

# ...Mutation Count
data_mc = data.dropna(subset=['Age at Diagnosis', 'Mutation Count'])  # exclude NAs
result_pearson = pearsonr(data_mc['Age at Diagnosis'], data_mc['Mutation Count'])
result_spearman = spearmanr(data_mc['Age at Diagnosis'], data_mc['Mutation Count'])
print('Age at Diagnosis and Mutation Count')
print('Pearson Correlation: ' + str(result_pearson.pvalue) + '\n'
      + 'Spearman Correlation: ' + str(result_spearman.pvalue))
print('---------------------------------------------------------')
# only spearman is significant

# ...Lymph nodes examined positive
data_ln = data.dropna(subset=['Age at Diagnosis', 'Lymph nodes examined positive'])  # exclude NAs
result_pearson = pearsonr(data_ln['Age at Diagnosis'], data_ln['Lymph nodes examined positive'])
result_spearman = spearmanr(data_ln['Age at Diagnosis'], data_ln['Lymph nodes examined positive'])
print('Age at Diagnosis and Lymph nodes examined positive')
print('Pearson Correlation: ' + str(result_pearson.pvalue) + '\n'
      + 'Spearman Correlation: ' + str(result_spearman.pvalue))
print('---------------------------------------------------------')
# none is significant

# ...Tumor Stage
data_ts = data.dropna(subset=['Tumor Stage'])  # exclude NAs

# Perform ANOVA
anova_result = f_oneway(*[data_ts['Age at Diagnosis'][data_ts['Tumor Stage'] == stage]
                          for stage in data_ts['Tumor Stage'].unique()])

print("ANOVA Result: " + str(anova_result.pvalue))  # significant


# ... ER Status
data_ts = data.dropna(subset=['ER Status'])  # exclude NAs
er_positive = data_ts[data_ts['ER Status'] == 1]
er_negative = data_ts[data_ts['ER Status'] == 0]

# Perform independent samples t-test
t_statistic, p_value = ttest_ind(er_positive['Age at Diagnosis'], er_negative['Age at Diagnosis'])

print("T-Test Result: " + str(p_value))  # significant


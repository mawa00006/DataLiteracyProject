import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from src.data_preprocessing import read_and_preprocess


# Read the data
data = pd.read_csv("../dat/preprocessed_brca_metabric_clinical_data.tsv", sep='\t')

### split Age at Diagnosis in categories ------------------------------
plt.hist(data['Age at Diagnosis'])
plt.xlabel('Age at Diagnosis')
plt.ylabel('Number of Patients')
plt.show()

# Age groups for 5 year but below 35 and above 85 will be in one group
bins = [0, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, float('inf')]  # Define the age bins
labels = ['<35', '35-40', '40-45', '45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '>=85']
data['Age Groups'] = pd.cut(data['Age at Diagnosis'], bins=bins, labels=labels, right=False)

sample_size_sqrt = np.sqrt(data.groupby('Age Groups').size()) / 20
sns.boxplot(x='Age Groups', y='Age at Diagnosis', data=data, width=sample_size_sqrt)
plt.show()


### Age vs Therapy ------------------------------




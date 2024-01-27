"""
@Author Friederike Moroff
@Author Mattes Warning
"""
import pandas as pd


def read_and_preprocess(data_path: str) -> pd.DataFrame:
    """
    Return the preprocessed METABRIC clinical dataset.
    :param data_path: A string containing the path to the METABRIC clinical dataset
    downloaded from https://www.cbioportal.org/study/summary?id=brca_metabric
    :return: A Dataframe containing the preprocessed data with shape (num_patients, num_features)
    """

    # Read the data
    data = pd.read_csv(data_path, sep='\t')

    # Only use sample from the METABRIC Nature 2012 publication
    data = data[data['Patient ID'].str.startswith('MB')]

    # Exclude redundant/identical valued columns
    exclude_col = ['Cancer Type', 'Study ID', 'Sample ID', 'Number of Samples Per Patient', 'Sample Type', 'Sex']
    data = data.loc[:, ~data.columns.isin(exclude_col)]

    # Drop NAs for relevant variables
    data = data.dropna(subset=['Age at Diagnosis', 'Chemotherapy', 'Tumor Size', 'Tumor Stage',
                                 'Neoplasm Histologic Grade', 'Lymph nodes examined positive',
                                 'Mutation Count', 'Integrative Cluster'])

    # Exclude samples with unspecific cancer type labeling
    data = data[~data['Cancer Type Detailed'].isin(["Breast"])]

    # Encode the binary variables as 0/1
    data['Chemotherapy Binary'] = (data['Chemotherapy'] == 'YES').astype(int)
    data['Type of Breast Surgery Binary'] = (data['Type of Breast Surgery'] == 'MASTECTOMY').astype(int)
    data['Hormone Therapy Binary'] = (data['Hormone Therapy'] == 'YES').astype(int)
    data['Radio Therapy Binary'] = (data['Radio Therapy'] == 'YES').astype(int)
    data['ER Status Binary'] = (data['ER Status'] == 'Positive').astype(int)

    # convert categorical to dummy numerical variable
    tumor_type_mapping = {'Breast Invasive Ductal Carcinoma': 0,
                          'Breast Mixed Ductal and Lobular Carcinoma': 1,
                          'Breast Invasive Lobular Carcinoma': 2,
                          'Invasive Breast Carcinoma': 3,
                          'Breast Invasive Mixed Mucinous Carcinoma': 4,
                          'Breast Angiosarcoma': 5,
                          'Metaplastic Breast Cancer': 6}

    cluster_mapping = {'1': 0, '2': 1, '3': 2, '4ER+': 3, '4ER-': 4, '5': 5,
                       '6': 6, '7': 7, '8': 8, '9': 9, '10': 10}

    data['Cancer Type Detailed Encoded'] = data['Cancer Type Detailed'].map(tumor_type_mapping)
    data['Integrative Cluster Encoded'] = data['Integrative Cluster'].map(cluster_mapping)

    return data


if __name__ == "__main__":
    file_name = "../dat/brca_metabric_clinical_data.tsv"
    metabric = read_and_preprocess(file_name)

    # Save preprocessed data
    metabric.to_csv("../dat/preprocessed_brca_metabric_clinical_data.tsv", sep='\t')


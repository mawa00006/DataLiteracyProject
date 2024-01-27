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

    # Drop NAs for Age as this is our primary variable of interest
    data = data.dropna(subset=['Age at Diagnosis'])

    # Exclude samples with unspecific cancer type labeling
    data = data[~data['Cancer Type Detailed'].isin(["Breast"])]

    # Encode the binary variables as 0/1
    data['Chemotherapy Binary'] = (data['Chemotherapy'] == 'YES').astype(int)
    data['Type of Breast Surgery Binary'] = (data['Type of Breast Surgery'] == 'MASTECTOMY').astype(int)
    data['Hormone Therapy Binary'] = (data['Hormone Therapy'] == 'YES').astype(int)
    data['Radio Therapy Binary'] = (data['Radio Therapy'] == 'YES').astype(int)
    data['ER Status Binary'] = (data['ER Status'] == 'Positive').astype(int)

    return data


if __name__ == "__main__":
    file_name = "../dat/brca_metabric_clinical_data.tsv"
    metabric = read_and_preprocess(file_name)

    # Save preprocessed data
    metabric.to_csv("../dat/preprocessed_brca_metabric_clinical_data.tsv", sep='\t')

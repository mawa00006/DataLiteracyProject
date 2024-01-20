import pandas as pd

def read_and_preprocess(file_name):
    # read the data ------------------------------
    data = pd.read_csv(file_name, sep='\t')

    # inspect the data ------------------------------
    # print(data.shape) # (2509, 39)
    # print(data.columns)

    # exclude columns that aren't needed -----
    #   Patient ID == Sample ID
    #   Columns that are the same for all:
    #   Cancer Type = 'Breast Cancer', Study ID = 'brca_metabric' , Number of Samples Per Patient = 1,
    #   Sample Type = 'Primary', Sex = 'Female'
    exclude_col = ['Cancer Type', 'Study ID', 'Sample ID', 'Number of Samples Per Patient', 'Sample Type', 'Sex']
    data = data.loc[:, ~data.columns.isin(exclude_col)]
    # print(data.shape) # (2509, 33)

    # interesting columns to keep -----
    #   Age at Diagnosis - numerical, until 2 decimal points
    #   Type of Breast Surgery - ['MASTECTOMY' 'BREAST CONSERVING' nan]
    #   Cancer Type Detailed - ['Breast Invasive Ductal Carcinoma'
    #       'Breast Mixed Ductal and Lobular Carcinoma'
    #       'Breast Invasive Lobular Carcinoma' 'Invasive Breast Carcinoma'
    #       'Breast Invasive Mixed Mucinous Carcinoma' 'Breast Angiosarcoma' 'Breast'
    #       'Metaplastic Breast Cancer']
    #       Cellularity - [nan 'High' 'Moderate' 'Low']
    #   Chemotherapy - ['NO' 'YES' nan]
    #   Pam50 + Claudin-low subtype - ['claudin-low' 'LumA' 'LumB' 'Normal' nan 'Her2' 'Basal' 'NC']
    #   Cohort - [ 1.  2.  3.  5.  4.  9.  7.  6. nan  8.]
    #   ER status measured by IHC - ['Positve' 'Negative' nan]
    #   ER Status - ['Positive' 'Negative' nan]
    #   Neoplasm Histologic Grade - [ 3.  2.  1. nan]
    #   HER2 status measured by SNP6 - ['NEUTRAL' 'LOSS' nan 'GAIN' 'UNDEF']
    #   HER2 Status - ['Negative' nan 'Positive']
    #   Tumor Other Histologic Subtype - ['Ductal/NST' 'Mixed' 'Lobular' 'Tubular/ cribriform' nan 'Mucinous'
    #       'Medullary' 'Other' 'Metaplastic']
    #   Hormone Therapy - ['YES' 'NO' nan]
    #   Inferred Menopausal State - ['Post' 'Pre' nan]
    #   Integrative Cluster - ['4ER+' '3' '9' '7' '4ER-' nan '5' '8' '10' '1' '2' '6']
    #   Primary Tumor Laterality - ['Right' 'Left' nan]
    #   Lymph nodes examined positive - numeric
    #   Mutation Count - numeric
    #   Nottingham prognostic index - numeric, three decimals
    #   Oncotree Code - ['IDC' 'MDLC' 'ILC' 'BRCA' 'IMMC' 'PBS' 'BREAST' 'MBC']
    #   Overall Survival (Months) - numeric
    #   Overall Survival Status - ['0:LIVING' '1:DECEASED' nan]
    #   PR Status - ['Negative' 'Positive' nan]
    #   Radio Therapy - ['YES' 'NO' nan]
    #   Relapse Free Status (Months) - numeric
    #   Relapse Free Status - ['0:Not Recurred' '1:Recurred' nan]
    #   3-Gene classifier subtype - ['ER-/HER2-' 'ER+/HER2- High Prolif' nan 'ER+/HER2- Low Prolif' 'HER2+']
    #   TMB (nonsynonymous) - numeric
    #   Tumor Size - numeric
    #   Tumor Stage - [ 2.  1.  4.  3.  0. nan]
    #   Patient's Vital Status - ['Living' 'Died of Disease' 'Died of Other Causes' nan]

    # data preprocessing ------------------------------
    # drop NAs for Age
    data = data.dropna(subset=['Age at Diagnosis'])
    # print(data.shape) # (2498, 33)

    # Cancer Type Detailed
    # Breast Invasive Ductal Carcinoma             1865
    # Breast Mixed Ductal and Lobular Carcinoma     269
    # Breast Invasive Lobular Carcinoma             192
    # Invasive Breast Carcinoma                     133
    # -------------------------------------------------
    # Breast Invasive Mixed Mucinous Carcinoma       25
    # Breast                                         21
    # Breast Angiosarcoma                             2
    # Metaplastic Breast Cancer                       2

    # exclude groups with too small sample size
    exclude_cancer_groups = ['Breast Invasive Mixed Mucinous Carcinoma','Breast',
                             'Breast Angiosarcoma','Metaplastic Breast Cancer']
    data = data[~data['Cancer Type Detailed'].isin(exclude_cancer_groups)]
    # print(data.shape) # (2448, 33)

    # convert the binary variables to 0 and 1
    data['Chemotherapy'] = (data['Chemotherapy'] == 'YES').astype(int)
    data['Type of Breast Surgery'] = (data['Type of Breast Surgery'] == 'MASTECTOMY').astype(int)
    data['Hormone Therapy'] = (data['Hormone Therapy'] == 'YES').astype(int)
    data['Radio Therapy'] = (data['Radio Therapy'] == 'YES').astype(int)
    data['ER Status'] = (data['ER Status'] == 'Positive').astype(int)

    # only use the first dataset MB
    filtered_df = data[~data['Patient ID'].str.startswith('MTS')]
    #print(filtered_df.shape)

    return filtered_df

if __name__ == "__main__":
    file_name = "/Users/friederikemoroff/Documents/university/bioinformatic/data_literacy/breast_cancer/clinical_data.tsv"
    data = read_and_preprocess(file_name)

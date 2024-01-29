import pandas as pd
from lifelines import KaplanMeierFitter


def KaplanMeier(df: pd.DataFrame, tumor_stage: int, variable: str):
    """
   Calculates the Kaplan-Meier between age groups <50 and >=50 for a given tumor stage for relapse free status or survival status.

    :param df: A pandas Dataframe.
    :param tumor_stage: An integer between 1-4 indicating the tumor stage of interest .
    :param variable: Either 'Relapse Free Status' or 'Overall Survival'.
    :return:
    kmf1: Kaplan-Meier estimator for age group >= 50 Years.
    kmf2: Kaplan-Meier estimator for age group > 50 Years.
    """

    # Get age groups
    group1 = df[(df['Chemotherapy Binary'] == 1) &
                (df['Age at Diagnosis'] >= 50) &
                (df['Tumor Stage'] == tumor_stage)]

    group2 = df[(df['Chemotherapy Binary'] == 1) &
                (df['Age at Diagnosis'] < 50) &
                (df['Tumor Stage'] == tumor_stage)]

    # Create Kaplan-Meier estimators for relapse for each group
    kmf_group1 = KaplanMeierFitter()
    kmf_group2 = KaplanMeierFitter()

    # Fit the Kaplan-Meier curves for each group
    kmf_group1.fit(durations=group1[variable + ' (Months)'], event_observed=group1[variable],
                   label='Tumor Stage ' + str(tumor_stage) + ', Age $\ge$ 50')
    kmf_group2.fit(durations=group2[variable + ' (Months)'], event_observed=group2[variable],
                   label='Tumor Stage ' + str(tumor_stage) + ', Age $<$ 50')

    return kmf_group1, kmf_group2

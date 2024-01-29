import pandas as pd
from lifelines.statistics import logrank_test


def LogRankTest(df: pd.DataFrame, tumor_stage: int, variable: str) -> [float, float]:
    """
    Performs a logrank test between age groups <50 and >=50 for a given tumor stage for relapse free status or
    survival status.

    :param df: A pandas Dataframe.
    :param tumor_stage: An integer between 1-4 indicating the tumor stage of interest .
    :param variable: Either 'Relapse Free Status' or 'Overall Survival'.
    :return:
    test_statistic: The test statistic of the logrank test.
    p_value: The p_value of the logrank test.
    """

    # Get age groups
    group1 = df[(df['Chemotherapy Binary'] == 1) &
                (df['Age at Diagnosis'] >= 50) &
                (df['Tumor Stage'] == tumor_stage)]

    group2 = df[(df['Chemotherapy Binary'] == 1) &
                (df['Age at Diagnosis'] < 50) &
                (df['Tumor Stage'] == tumor_stage)]

    # Perform log-rank test
    results = logrank_test(group1[variable + ' (Months)'], group2[variable + ' (Months)'],
                           event_observed_A=group1[variable], event_observed_B=group2[variable])

    return results.test_statistic, results.p_value
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.multitest import multipletests
from src.data_preprocessing import read_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.metrics import f1_score as F1_score

from matplotlib import pyplot as plt


def LogisticRegression(data: pd.DataFrame, dep_variable: str, indep_variables: list, show_summary: bool = False):
    """
    Performs logistic regression.
    :param data: A Dataframe
    :param dep_variable: Name of the dependent variable
    :param indep_variables: List with at least one name of an independent variable
    :param show_summary: Default: False. Boolean values indication whether to print the model summary or not.
    :return:

    """
    # splitting into training and test set and
    X_train, X_test, Y_train, Y_test = train_test_split(data[indep_variables],
                                                        data[dep_variable],
                                                        test_size=0.25,
                                                        random_state=42)

    # building the model and fitting the data
    log_reg = sm.Logit(Y_train, X_train).fit()

    # print model summary
    if show_summary:
        print(log_reg.summary())

    pseudo_R_squared = log_reg.prsquared
    p_values = log_reg.pvalues

    # correcting for multiple testing
    if len(indep_variables) > 1:
        reject, adjusted_p_values, _, _ = multipletests(log_reg.pvalues, alpha=0.05, method='bonferroni')

    # performing predictions on the test dataset
    pred = log_reg.predict(X_test)
    predictions = list(map(round, pred))

    # confusion matrix
    conf_matrix = confusion_matrix(Y_test, predictions)
    conf_matrix = pd.DataFrame(conf_matrix, index=['True 0', 'True 1'], columns=['Pred 0', 'Pred 1'])

    # accuracy score
    test_accuracy = accuracy_score(Y_test, predictions)

    # ROC AUC
    fpr, tpr, thresholds = roc_curve(Y_test, predictions)
    roc_auc = auc(fpr, tpr)

    return p_values, round(test_accuracy, 2), round(roc_auc, 2), conf_matrix, round(pseudo_R_squared, 3)

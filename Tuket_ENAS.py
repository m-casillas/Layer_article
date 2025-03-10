import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats

def Tukey_ENAS(data):

    data = pd.read_csv(data)

    group_column = 'group_column'
    value_column = 'value_column'

    tukey_result = pairwise_tukeyhsd(data[value_column], data[group_column], alpha=0.05)

    return tukey_result.summary()

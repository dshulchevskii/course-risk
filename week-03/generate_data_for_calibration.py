import pandas as pd
import numpy as np
from scipy.special import expit


def make_category_feature(df):
    np.random.seed(42)
    ind_1 = df.query('default_flg == 1').index
    ind_0 = df.query('default_flg == 0').index

    values = ['A', 'B', 'C', 'D', 'E']
    prob_1 = [0.15, 0.12, 0.32, 0.10, 0.31]
    prob_0 = [0.15, 0.19, 0.25, 0.13, 0.28]

    feature = pd.Series(index=df.index)

    feature[ind_1] = np.random.choice(values, size=len(ind_1), p=prob_1)
    feature[ind_0] = np.random.choice(values, size=len(ind_0), p=prob_0)
    return feature


def generate_data(N=1000):

    np.random.seed(42)

    return (
        pd.DataFrame({'logit': np.random.randn(N) - 2})
        .assign(pd_true=lambda x: expit(x['logit']),
                pd=lambda x: expit(0.5 * x['logit'] - 1))
        .assign(default_flg=lambda x: np.random.binomial(1, x['pd_true']))
        .assign(category_feature=make_category_feature)
        .drop(columns=['pd_true', 'logit'])
    )


df = generate_data(N=30000)
df.to_csv('calibration.csv', index=False)

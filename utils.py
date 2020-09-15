import pandas as pd
from sklearn.preprocessing import minmax_scale


def pandas_minmax(df, axis):
    return pd.DataFrame(minmax_scale(df, axis=axis), index=df.index, columns=df.columns)

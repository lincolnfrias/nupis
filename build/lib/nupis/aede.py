def normalize(x):
    x_norm = (x - x.mean()) / x.std()
    return x_norm

def moran_summary(im):
    im = im
    print( 'Moran Rate Summary Report')
    print( '=========================')
    print( 'I       {}   observed value of Moran’s I'.format("%6.3f" % im.I))
    print( 'EI_sim   {}   average value of I from permutations  '.format("%6.3f" % im.EI_sim))
    print( 'p_sim    {}   p-value based on permutations'.format("%6.3f" % im.p_sim))

def moran_df(df, w):

    import pandas as pd
    import pysal as ps

    df = df.select_dtypes(include=['int64', 'float64'])
    lista1 = []
    lista2 = []

    for i in df.columns:
        x = ps.Moran(df[i], w)
        lista2.append(x.I)
        lista1.append(i)

    df = pd.DataFrame({'variavel':lista1, 'valor':lista2})
    df = df[['variavel', 'valor']]
    df = df.sort_values(by='valor', ascending=False)

    return df

def moran_plot(IM):

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pysal as ps

    y_norm = normalize(IM.y)
    y_lag = ps.lag_spatial(IM.w, IM.y)
    y_lag_norm = normalize(y_lag)
    dados = pd.DataFrame({'y':IM.y, 'y_norm':y_norm,
                          'y_lag':y_lag, 'y_lag_norm':y_lag_norm})

    f, ax = plt.subplots(1, figsize=(7, 5))
    sns.regplot('y_norm', 'y_lag_norm', data=dados, ci=None,
                color='black', line_kws={'color':'red'})
    plt.axvline(0, c='gray', alpha=0.7)
    plt.axhline(0, c='gray', alpha=0.7)

    limits = np.array([y_norm.min(), y_norm.max(), y_lag_norm.min(), y_lag_norm.max()])
    limits = np.abs(limits).max()
    border = 0.02
    ax.set_xlim(- limits - border, limits + border)
    ax.set_ylim(- limits - border, limits + border)

    plt.show();

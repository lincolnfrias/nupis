import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gp
import pysal as ps
from pysal.contrib.viz import mapping as maps
import palettable

def simplificar_nomes(df, variavel):
    '''
    Retira acentos, substitui espaços e
    hífens por underline e passa para caixa baixa.
    Não serve para simplicar nomes de colunas,
    use a função simplificar_colunas()

    df: dataframe

    variavel: nome da variável a transformar, entre aspas.

    '''

    df[variavel] = (df[variavel].str.strip()
                    .str.lower()
                    .str.replace(' ', '_')
                    .str.replace('\'', '')
                    .str.replace('-','_')
                    .str.replace('á', 'a')
                    .str.replace('é', 'e')
                    .str.replace('í', 'i')
                    .str.replace('ó', 'o')
                    .str.replace('ú', 'u')
                    .str.replace('â', 'a')
                    .str.replace('ê', 'e')
                    .str.replace('ô', 'o')
                    .str.replace('í', 'i')
                    .str.replace('ã','a')
                    .str.replace('õ','o')
                    .str.replace('ç','c')
                    .str.replace('à', 'a'))

def simplificar_colunas(df):
    '''
    Retira acentos, substitui espaços e
    hífens por underline e passa para caixa baixa.

    df: dataframe

    '''

    df.columns = (df.columns.str.strip()
                    .str.lower()
                    .str.replace(' ', '_')
                    .str.replace('\'', '')
                    .str.replace('-','_')
                    .str.replace('á', 'a')
                    .str.replace('é', 'e')
                    .str.replace('í', 'i')
                    .str.replace('ó', 'o')
                    .str.replace('ú', 'u')
                    .str.replace('â', 'a')
                    .str.replace('ê', 'e')
                    .str.replace('ô', 'o')
                    .str.replace('í', 'i')
                    .str.replace('ã','a')
                    .str.replace('õ','o')
                    .str.replace('ç','c')
                    .str.replace('à', 'a'))

def normalizar(x):
    x_norm = (x - x.mean()) / x.std()
    return x_norm

def mesclar_shp_df(shp, df):
    geodf = gp.read_file(shp)
    geodf.geometry = geodf.geometry.simplify(0.001)
    geodf.rename(columns={'CD_GEOCMU': 'mun'}, inplace=True)
    geodf['mun'] = geodf.mun.astype(int)
    df = pd.read_csv(df)
    name = pd.merge(geodf, df, on='mun', suffixes=('', '_y'))
    return name

def mapa(df, variavel, scheme='equal_interval', cmap='Set1', k=4):

    ax = df.plot(column=variavel, scheme=scheme, k=k,
             linewidth=0, figsize=(7,7), legend=True, cmap=cmap)
    ax.set_axis_off();


def moran_resumo(im):
    im = im
    print( 'Moran Rate Summary Report')
    print( '=========================')
    print( 'I       {}   observed value of Moran’s I'.format("%6.3f" % im.I))
    print( 'EI_sim   {}   average value of I from permutations  '.format("%6.3f" % im.EI_sim))
    print( 'p_sim    {}   p-value based on permutations'.format("%6.3f" % im.p_sim))

def moran_df(df, w):

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

def moran_dispersao(IM, title='', xlabel='', ylabel=''):

    y_norm = normalizar(IM.y)
    y_lag = ps.lag_spatial(IM.w, IM.y)
    y_lag_norm = normalizar(y_lag)
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
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show();

def lisa_mapa(variavel, shapefile, p_thres=0.05, **kws):

    w = ps.queen_from_shapefile(shapefile)
    lisa = ps.Moran_Local(variavel, w)

    fig = plt.figure(figsize=(9, 7))
    shp = ps.open(shapefile)
    base = maps.map_poly_shp(shp)
    base = maps.base_lisa_cluster(base, lisa, p_thres=p_thres)
    base.set_edgecolor('1')
    base.set_linewidth(0.1)
    ax = maps.setup_ax([base], [shp.bbox])

    boxes, labels = maps.lisa_legend_components(lisa, p_thres=p_thres)
    plt.legend(boxes, labels, fancybox=True, **kws)

    plt.show();

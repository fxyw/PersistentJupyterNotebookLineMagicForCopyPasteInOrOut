import pandas as pd
import numpy as np
import scipy as sp
#%matplotlib inline
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"]="FangSong"
mpl.rcParams['axes.unicode_minus']=False
import seaborn as sns
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)
#%load_ext autoreload
get_ipython().run_line_magic('load_ext', 'autoreload')
#%autoreload 2
get_ipython().run_line_magic('autoreload', '2')
#%automagic 1
get_ipython().run_line_magic('automagic', '1')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from pandas.api.types import is_numeric_dtype
def m_v(x):return np.mean(x)+np.nan_to_num(pd.Series(sp.stats.moment(x,2)).div(np.mean(x)-np.min(x))[0])
def m_s(x):return np.mean(x)+np.nan_to_num(pd.Series(sp.stats.moment(x,3)).div(sp.stats.moment(x,2))[0])
def m_k(x):return np.min(x)+np.nan_to_num(np.sqrt(pd.Series(sp.stats.moment(x,4)).div(sp.stats.moment(x,2))[0]))
def spvar(x):return sp.stats.moment(x.dropna(),2)
def esf2(x):return np.nan_to_num(pd.Series(sp.stats.kurtosis(x)).divide(sp.stats.skew(x)**2)[0] if sp.stats.skew(x)!=0 else np.nan)
def cv1(x):return np.nan_to_num(pd.Series(spstd(x)).div(np.mean(x))[0] if np.mean(x)!=0 else np.nan)
def sf1(x):return np.nan_to_num(sp.stats.moment(x,4)*sp.stats.moment(x,2)/sp.stats.moment(x,3)**2 if sp.stats.moment(x,3)!=0 else np.nan)
def msf(x):return np.nan_to_num(sp.stats.kurtosis(x,fisher=False) if np.abs(sp.stats.skew(x))<1 else pd.Series(sp.stats.kurtosis(x,fisher=False)).divide(sp.stats.skew(x)**2)[0])
def spstd(x):return np.sqrt(sp.stats.moment(x.dropna(),2)) if is_numeric_dtype(x) else np.nan
def hdmedian1(x):return np.nan_to_num(sp.stats.mstats.hdmedian(x) if np.max(x)>np.min(x) else np.min(x))
def n_c(x):return np.sqrt(len(x))
def n_s(x):return np.nan_to_num(pd.Series(sp.stats.moment(x,3)).div(sp.stats.moment(x,2)).div(np.max(x)-np.mean(x))[0])
def n_k(x):return np.nan_to_num(np.sqrt(pd.Series(sp.stats.moment(x,4)).div(sp.stats.moment(x,2))).div(np.max(x)-np.min(x))[0])
def s_k(x):return np.sqrt(sp.stats.kurtosis(x,fisher=False))
def s_m(x):return np.sqrt(np.max(x)) if np.max(x)>0 else -np.sqrt(-np.max(x))
def nsf(x):return np.nan_to_num(np.sqrt(sp.stats.kurtosis(x,fisher=False))*(-1 if sp.stats.skew(x)<0 else 1) if np.abs(sp.stats.skew(x))<1 else pd.Series(np.sqrt(sp.stats.kurtosis(x,fisher=False))).divide(sp.stats.skew(x))[0])
def s_f(x):return np.nan_to_num(pd.Series(np.sqrt(sp.stats.kurtosis(x,fisher=False))).divide(sp.stats.skew(x))[0] if sp.stats.skew(x)!=0 else np.nan)
def mad(x):return np.nan_to_num(pd.Series(sp.stats.median_abs_deviation(x,scale='normal')).div(np.median(x))[0] if np.median(x)!=0 else np.nan)
def mad1(x):return np.nan_to_num(pd.Series(sp.stats.median_abs_deviation(x,scale='normal')).div(sp.stats.mstats.hdmedian(x).data.tolist())[0] if np.max(x)>np.min(x) else np.nan)
def mad2(x):return np.nan_to_num(pd.Series(sp.stats.median_abs_deviation(x,scale='normal')).div(hdmedian1(x))[0] if hdmedian1(x)!=0 else np.nan)
def mad3(x):return np.nan_to_num(pd.Series(sp.stats.median_abs_deviation(x,scale='normal')).div(np.mean(x))[0] if np.mean(x)!=0 else np.nan)
def mad4(x):return np.nan_to_num(pd.Series(sp.stats.median_abs_deviation(x,scale='normal')).div(mod1(x))[0] if mod1(x)!=0 else np.nan)
def n_v(x):return np.nan_to_num(pd.Series(sp.stats.moment(x,2)).div(np.mean(x)-np.min(x)).div(np.max(x)-np.mean(x))[0])
def sef(x):return np.nan_to_num(pd.Series(np.sqrt(sp.stats.kurtosis(x)) if sp.stats.kurtosis(x)>0 else -np.sqrt(-sp.stats.kurtosis(x))).divide(sp.stats.skew(x))[0] if sp.stats.skew(x)!=0 else np.nan)
def smv(x):return np.sqrt(m_v(x)) if m_v(x)>0 else -np.sqrt(-m_v(x))
def sms(x):return np.sqrt(m_s(x)) if m_s(x)>0 else -np.sqrt(-m_s(x))
def smk(x):return np.sqrt(m_k(x)) if m_k(x)>0 else -np.sqrt(-m_k(x))
def cmv(x):return np.cbrt(m_v(x))
def cms(x):return np.cbrt(m_s(x))
def cmk(x):return np.cbrt(m_k(x))
def c_c(x):return np.cbrt(len(x))
def c_k(x):return np.cbrt(sp.stats.kurtosis(x,fisher=False))
def c_m(x):return np.cbrt(np.max(x))
import re
p = re.compile('([^a-z_1-9]+)([a-z_1-9]*)')
import itertools
from io import StringIO
import requests
def read_url(url):return pd.read_csv(StringIO(requests.get(url).text))
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100%; margin-left: 0%; margin-right:auto;}</style>"))
display(HTML("<style>div.cell { width:100%; margin-left: 0%; margin-right:auto;}</style>"))
import statsmodels.formula.api as sm
import statsmodels.api
import sklearn.linear_model
import lightgbm as lgbm
import shap
import xgboost
from sklearn.impute import SimpleImputer
from scipy.stats import mode
def mod1(x):return np.nan_to_num(mode(x)[0][0])
import copy
def r2saessecount(y,prd1):
    print([1 - ((y -prd1)**2).sum() / ((y -y.mean())**2).sum(), sum(list(map(abs,y-prd1))), sum(list(map(lambda x: np.power(x,2),y-prd1))), sum(list(map(lambda x: 1 if abs(x)>=0.5 else 0,y-prd1)))])
def fittoextreme(df, cols, target): return df.groupby(cols).apply(lambda group:group[target].count()-group[target].value_counts().max()).sum()
from patsy import ModelDesc
def bar_plot(shap_values, cols, h=12):
   import numpy as np
   import pandas as pd
   import seaborn as sns    
   from matplotlib import rcParams
   rcParams['figure.figsize'] = 12, h
   rcParams['font.size'] = 11
   df=pd.DataFrame({'x':cols, 'y':np.mean(np.abs(shap_values),0).tolist()})
   splot=sns.barplot(y='x', x='y', data=df, order=df.sort_values('y', ascending = False).x.to_list(), color='blue')
   for p in splot.patches:
       splot.annotate(format(p.get_width(), '.3f'), 
                      (p.get_width(),p.get_y() + p.get_height() / 2.), 
                      ha = 'center', va = 'center', 
                      xytext = (15, 0), 
                      textcoords = 'offset points') 
#mic tic dcor rdc
from minepy import MINE
m = MINE()
def mic(x,y):
    m.compute_score(x, y)
    return m.mic()
def tic(x,y):
    m.compute_score(x, y)
    return m.tic(norm=True)
from scipy.spatial.distance import pdist, squareform
def dcor(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    return np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
from scipy.stats import rankdata
def rdc(x, y, f=np.sin, k=20, s=1/6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)
    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))
    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)
    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])
    # Random linear projections
    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)
    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)
    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)
    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:
        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]
        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))
        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2
    return np.sqrt(np.max(eigs))
from sklearn.feature_selection import mutual_info_regression
def mir(x,y): return mutual_info_regression(x.reshape(-1,1), y)[0]

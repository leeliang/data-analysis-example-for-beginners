import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['STHeiti']
mpl.rcParams['font.serif'] = ['STHeiti']
import seaborn as sns
sns.set_style("darkgrid",{"font.sans-serif":['STHeiti', 'STHeiti']})
import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import metrics
fp = pd.read_csv('data/fund_performance.csv',dtype={'code':str})
fp.set_index('code',inplace=True)
nav = pd.read_csv('data/two_years_data.csv')
nav['date'] = pd.to_datetime(nav['date'])
nav.set_index('date',inplace=True)
rate_range = []
for p in [0.0,0.1,0.2,0.4,0.6,0.8,0.9,1.0]:
    rate_range.append(nav.unstack().quantile(p))
rate_level = nav.apply(lambda x: pd.cut(x,rate_range,labels=np.arange(7)))
def calLevel(x):
    level_nums = []
    for level in np.arange(7):
        level_nums.append((x==level).sum())
    return level_nums
level = rate_level.apply(lambda x: calLevel(x))
df_level = level.apply(lambda x: pd.Series(x))
feature = pd.merge(df_level,fp,how='left',right_index=True,left_index=True)
feature.rename(index=str, columns={0: "level_0", 1: "level_1", 2: "level_2",
                                   3: "level_3", 4: "level_4", 5: "level_5",
                                   6: "level_6"},inplace=True)
feature.head()
f_scaler = feature.apply(preprocessing.minmax_scale)
data = f_scaler.values
ns = np.arange(2,10)
silhouette_score = []
sse = []
for n in ns:
    km = KMeans(n_clusters=n)
    label = km.fit_predict(data)
    silhouette_score.append(metrics.silhouette_score(data,label,metric='euclidean'))
    sse.append(km.inertia_)
fig, host = plt.subplots()
l1, = host.plot(ns, sse, 'b--',marker='o')
host.set_xlabel('Clustering Number')
ax2 = host.twinx()
l2, = ax2.plot(ns, silhouette_score, 'r-.',marker='o')
ax2.tick_params('y', colors='r')
lines = [l1,l2]
host.legend(lines,[u'SSE', u'Silhouette coefficient'])
km = KMeans(n_clusters=3)
label = km.fit_predict(data)
feature['label'] = label
sns.pairplot(vars=['return','beta','r_squared'],data=feature,hue='label',size=3)
def dis_ten(fea):
    """fea: Series"""
    fea_range = [fea.quantile(x) for x in np.arange(11)/10.]
    fea_range[0] = fea_range[0] - 0.1
    fea_range = set(fea_range)
    fea_range = np.sort(list(fea_range))
    return pd.cut(fea,fea_range,labels=np.arange(len(fea_range)-1))
f_scaler = feature.apply(lambda x: dis_ten(x))
data = f_scaler.values
ns = np.arange(2,10)
silhouette_score = []
sse = []
for n in ns:
    km = KMeans(n_clusters=n)
    label = km.fit_predict(data)
    silhouette_score.append(metrics.silhouette_score(data,label,metric='euclidean'))
    sse.append(km.inertia_)
fig, host = plt.subplots()
l1, = host.plot(ns, sse, 'b--',marker='o')
host.set_xlabel('Clustering Number')
ax2 = host.twinx()
l2, = ax2.plot(ns, silhouette_score, 'r-.',marker='o')
ax2.tick_params('y', colors='r')
lines = [l1,l2]
host.legend(lines,[u'SSE', u'Silhouette coefficient'])
km = KMeans(n_clusters=4)
label = km.fit_predict(data)
feature['label'] = label
sns.pairplot(vars=['return','beta','r_squared'],data=feature,hue='label',size=3)
sns.pairplot(vars=['return','level_0','level_1','level_2','level_3','level_4','level_5'],data=feature,hue='label',size=3)

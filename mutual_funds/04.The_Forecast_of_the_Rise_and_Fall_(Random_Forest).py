import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import preprocessing
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
rate = pd.read_csv('data/two_years_data.csv')
rate['date'] = pd.to_datetime(rate['date'])
rate.set_index('date',inplace=True)
fp = pd.read_csv('data/fund_performance.csv',dtype={'code':str})
fp.set_index('code',inplace=True)
inf = pd.read_csv('./data/mixed_funds.csv',dtype={'code':str})
inf.set_index('code',inplace=True)
fpi = pd.merge(fp,inf,left_index=True,right_index=True)
fpi.head()
manger_dict = fpi.groupby('manager')['return'].mean().to_dict()
fpi['manager'] = fpi['manager'].replace(manger_dict)
company_dict = fpi.groupby('company')['return'].mean().to_dict()
fpi['company'] = fpi['company'].replace(company_dict)
fpi.drop(['name','type'],axis=1,inplace=True)
rate_range = [rate.unstack().quantile(x) for x in np.arange(11)/10.]
rate_range[0] = rate_range[0] - 0.1
rate_range = np.sort(list(set(rate_range)))
n = len(rate_range)
rate_level = rate.apply(lambda x: pd.cut(x,rate_range,labels=np.arange(n-1)))
rate_range
time_range = pd.date_range('2016-01',periods=22,freq='M').strftime('%Y-%m')
month_rate = ((rate/100+1).cumprod().ix[-1].mean()-1)*250/len(rate)/12
fea = pd.DataFrame()
for i in range(len(time_range)-7):
    s_time = time_range[i]
    e_time = time_range[i+5]
    level_nums = []
    for level in np.arange(n-1):
        level_nums.append((rate_level[s_time:e_time]==level).sum())
    col_labels = ['L'+str(x) for x in np.arange(n-1)]
    df_level = pd.DataFrame(level_nums).T
    df_level.columns = col_labels
    label = (rate[time_range[i+3]]/100+1).cumprod().ix[-1]>month_rate+1
    label.name = 'label'
    df_merge = pd.concat([df_level,fpi,label],axis=1).reset_index()
    df_merge.rename(index=str, columns={},inplace=True)
    fea = pd.concat([fea,df_merge])
fea.reset_index()
fea.drop('index',inplace=True)
month_rate*12
X = fea.drop(['code','label'],axis=1)
def dis_ten(fea):
    """fea: Series"""
    fea_range = [fea.quantile(x) for x in np.arange(11)/10.]
    fea_range[0] = fea_range[0] - 0.1
    fea_range = set(fea_range)
    fea_range = np.sort(list(fea_range))
    return pd.cut(fea,fea_range,labels=np.arange(len(fea_range)-1))
X_scaler = X.apply(lambda x: dis_ten(x))
y = fea['label']
no_trees = range(1,30)
k_fold=10
df_random_forest = pd.DataFrame()
for i in no_trees:    
    forest = RandomForestClassifier(n_estimators=i)
    scores = cross_validation.cross_val_score(forest, X_scaler, y,scoring='f1', cv=k_fold)
    df_random_forest[str(i)]=scores
plt.figure(figsize=(16,9))
df_random_forest.boxplot()
plt.xlabel( u"树数量", fontsize=14)
plt.ylabel(u"准确率", fontsize=14)
sns.set_context('poster')
forest = RandomForestClassifier(n_estimators=25)
clf = forest.fit(X_scaler,y)
index = np.arange(len(X_scaler.columns))
plt.figure(figsize=(10,8))
plt.bar(index, clf.feature_importances_, 0.35,label='')
plt.xlabel('属性', fontsize =16)
plt.ylabel('属性重要性', fontsize =16)
plt.xticks(index, X_scaler.columns,rotation = 70)

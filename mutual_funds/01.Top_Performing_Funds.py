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
rate = pd.read_csv('./data/mixed_funds_rate.csv')
rate['date'] = pd.to_datetime(rate['date'])
rate.set_index('date',inplace=True)
rate.head()
years = np.arange(2001,2018)
funds_nums_with_data = []
for year in years:
    data = rate[str(year)]
    max_days_with_data = data.notnull().sum().max()
    funds_nums_with_data.append((data.notnull().sum() == max_days_with_data).sum())
year_count = pd.DataFrame()
year_count['nums'] = funds_nums_with_data
year_count['year'] = years.astype(str)
year_count['year'] = pd.to_datetime(year_count['year'])
year_count.set_index('year',inplace=True)
ax = year_count.plot(kind='bar')
xtl=[item.get_text()[:4] for item in ax.get_xticklabels()]
ax.set_xticklabels(xtl)
plt.gcf().autofmt_xdate()
df = rate['2016':'2017'].dropna(how='all')
df.notnull().sum().plot(ls='None',marker='.')
df400 = df[df.columns[df.notnull().sum()>400]].fillna(0)
df400.sort_index(ascending=True,inplace=True)
df400.to_csv('data/two_years_data.csv')
rm = df400.mean(axis=1)
returns_mean = (rm/100+1).cumprod()-1
hs300 = pd.read_csv('./data/hs300.csv',header=None,names={'date','hs300'})
hs300['date'] = pd.to_datetime(hs300['date'])
hs300.set_index('date',inplace=True)
hs300.sort_index(ascending=True,inplace=True)
hs_returns = (hs300/100+1).cumprod()-1
hs_returns.plot(label=u'沪深300')
returns_mean.plot(label=u'混合型基金平均表现')
plt.legend()
returns = ((df400/100+1).cumprod().ix[-1]-1)*250/len(df400)
beta = {}
for code in df400.columns:
    rho = rm.corr(df400[code])
    beta[code] = rho*df400[code].std()/rm.std()    
beta = pd.Series(beta)
R_m = returns_mean.ix[-1]*250/len(df400)
r_f = 0.03
alpha = returns-r_f - pd.Series(beta)* (R_m-r_f)
r_squared = {}
for code in df400.columns:
    r_squared[code] = rm.corr(df400[code])**2*100
r_squared = pd.Series(r_squared)
std = df400.std()
fund_performance = returns.to_frame()
fund_performance.columns = ['return']
fund_performance['beta'] = beta
fund_performance['alpha'] = alpha
fund_performance['r_squared'] = r_squared
fund_performance['std'] = std
fund_performance.hist(bins=100,figsize=(10,6),layout=[2,3])
fund_performance.corr()
fp = fund_performance
c_1 = fp['beta'] >1
c_2 = (fp['r_squared']>=80) & fp['r_squared']<=100
c_3 = fp['return']>fp['return'].quantile(0.9)
c_4 = fp['alpha']>fp['alpha'].quantile(0.9)
good = c_1 & c_2 & c_3 & c_4 
good_funds = fp[good]
(df400/100+1).cumprod().T[good].T.plot(legend=False)
fp['up_days'] = (df400.T>0).sum(axis=1)
fp.corr()
ax = fp['up_days'].plot(kind='hist',bins=100,label=u'所有基金',normed=True)
fp[good]['up_days'].plot(ax=ax,kind='hist',bins=20,label=u'挑选基金',normed=True,alpha=0.5)
ax.legend()
ax.set_title(u'每只基金上涨天数分布')
fp['max_rate'] = df400.max()
fp['min_rate'] = df400.min()
fp.corr()
fig, axes = plt.subplots(nrows=1, ncols=2)
fp['max_rate'].plot(ax=axes[0],kind='hist',bins=100,label=u'所有基金',normed=True)
fp[good]['max_rate'].plot(ax=axes[0],kind='hist',bins=20,label=u'优选基金',normed=True,alpha=0.5)
axes[0].set_xlim(0,10)
axes[0].legend()
axes[0].set_title(u'最大涨幅分布')
fp['min_rate'].plot(ax=axes[1],kind='hist',bins=100,label=u'所有基金',normed=True)
fp[good]['min_rate'].plot(ax=axes[1],kind='hist',bins=20,label=u'优选基金',normed=True,alpha=0.5)
axes[1].set_xlim(-20,0)
axes[1].set_title(u'最大跌幅分布')
top1 = fp[good]['up_days'] > fp[good]['up_days'].quantile(0.5)
top2 = fp[good]['max_rate'] > fp[good]['max_rate'].quantile(0.5)
top3 = fp[good]['return'] > fp[good]['return'].quantile(0.5)
top = top1 & top2 & top3
inf = pd.read_csv('./data/mixed_funds.csv',dtype={'code':str})
inf.set_index('code',inplace=True)
inf_top = inf[inf.index.isin(fp[good][top].index)]
top_funds = pd.merge(fp[good][top],inf_top,left_index=True,right_index=True)
top_funds.sort_values(by='return',ascending=False)
fp.index.name = 'code'
fp.to_csv('./data/fund_performance.csv')

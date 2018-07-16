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
rate = pd.read_csv('./data/mixed_funds_rate.csv',dtype={'code':str})
rate['date'] = pd.to_datetime(rate['date'])
rate.set_index('date',inplace=True)
rate.sort_index(inplace=True)
inf = pd.read_csv('./data/mixed_funds.csv',dtype={'code':str})
inf.set_index('code',inplace=True)
years = range(2015,2018)
rate_per_year = pd.DataFrame(index=rate.columns)
for year in years:
    df = rate[str(year)]
    df =df[df.columns[df.notnull().sum()>200]]
    rate_per_year[str(year)] = (df/100+1).cumprod().ix[-1]-1
rate_per_year.dropna(inplace=True)
df = pd.merge(rate_per_year,inf,how='left',right_index=True,left_index=True)
df.insert(3,'mean',rate_per_year.mean(axis=1))
df.insert(3,'std',rate_per_year.std(axis=1))
df.head()
df.groupby('manager')['mean'].mean().sort_values(ascending=False)[:20].plot(kind='barh')
plt.savefig("/Users/lli/Documents/GitHub/zhihu/source/img/funds/3_2.png",dpi=300)
df[df['manager']==u"魏伟"]
rate_per_year.std(axis=1).plot(kind='hist',bins=50)
plt.savefig("/Users/lli/Documents/GitHub/zhihu/source/img/funds/3_4.png",dpi=300)
df[df['std']<0.6].groupby('manager')['mean'].mean().sort_values(ascending=False)[:20].plot(kind='barh')
plt.savefig("/Users/lli/Documents/GitHub/zhihu/source/img/funds/3_5.png",dpi=300)
df[df['manager']==u"杨飞"]
(rate['2015':'2017'][['100056','020003','160211','160212']]/100+1).cumprod().plot()
plt.savefig("/Users/lli/Documents/GitHub/zhihu/source/img/funds/3_7.png",dpi=300)
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,7))
df.groupby('company')['mean'].mean().sort_values(ascending=False)[:10].plot(kind='barh',ax=axes[0][0])
axes[0,0].set_title(u"收益排名前十公司")
df[df['std']<0.6].groupby('company')['mean'].mean().sort_values(ascending=False)[:10].plot(kind='barh',ax=axes[0][1])
axes[0,1].set_title(u"收益排名前十公司 (std < 0.6)")
top10 = df.groupby('company')['mean'].mean().sort_values(ascending=False)[:10].index
df[df['company'].isin(top10)]['company'].value_counts().plot(kind='barh',ax=axes[1,0])
axes[1,0].set_title(u"收益排名前十公司管理基金数量")
top10 = df[df['std']<0.6].groupby('company')['mean'].mean().sort_values(ascending=False)[:10].index
df[df['company'].isin(top10)]['company'].value_counts().plot(kind='barh',ax=axes[1,1])
axes[1,1].set_title(u"收益排名前十公司管理基金数量 (std<0.6)")
plt.tight_layout()
plt.savefig("/Users/lli/Documents/GitHub/zhihu/source/img/funds/3_8.png",dpi=300)
fund  = rate['160212'].dropna().to_frame()
fund['weekday'] = fund.index.dayofweek+1
fund.groupby('weekday').mean().plot(kind='bar')
plt.savefig("/Users/lli/Documents/GitHub/zhihu/source/img/funds/3_9.png",dpi=300)
fund_nav = (fund['160212']/100.+1).cumprod().to_frame()
fund_nav['weekday'] = fund_nav.index.dayofweek + 1
principal = 1
sale = True
buy = False
fee = 0.1/100
for day in range(len(fund_nav)):
    if sale & (fund_nav['weekday'][day]==4):
        share = principal*(1-fee)/fund_nav['160212'][day]
        sale = False
        buy = True
    if buy & (fund_nav['weekday'][day]==3):
        principal = share*fund_nav['160212'][day]*(1-fee)
        sale = True
        buy = False
print u"是时候展示操作的收益： %f"  % principal
print u"不操作收益： %f"  % fund_nav['160212'][-1]
fund['day_in_month'] = fund.index.day
fund.groupby('day_in_month')['160212'].mean().plot(kind='bar')
plt.savefig("/Users/lli/Documents/GitHub/zhihu/source/img/funds/3_10.png",dpi=300)
fund['month'] = fund.index.month
fund.groupby('month')['160212'].mean().plot(kind='bar')
plt.savefig("/Users/lli/Documents/GitHub/zhihu/source/img/funds/3_11.png",dpi=300)

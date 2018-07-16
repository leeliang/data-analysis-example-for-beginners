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
funds = pd.read_csv('./data/funds.csv',dtype={'code':str})
funds.head()
funds.describe()
funds['type'].value_counts().plot(kind='barh',figsize=(10,8))
mixed_funds = funds[funds['type']==u"混合型"]
mixed_funds.describe()
mixed_funds['manager'].value_counts()[:10].plot('barh')
mixed_funds['company'].value_counts()[:10].plot(kind='barh')
mixed_funds.to_csv('./data/mixed_funds.csv',index=False)

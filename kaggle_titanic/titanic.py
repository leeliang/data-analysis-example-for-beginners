#!/usr/bin/env python
# coding: utf-8
"""
Created on 2017-08
@author: Liang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  cross_validation, tree, preprocessing, metrics
from IPython.display import Image 
import pydotplus
#
train_data = pd.read_csv('./data/train.csv')

plt.subplots(nrows=2, ncols=2)
plt.suptitle("Ratio of Survived")
plt.subplot(221)
train_data.groupby('Pclass').mean()['Survived'].plot.bar(alpha=0.5)


plt.subplot(222)
train_data.groupby(['Sex']).mean()['Survived'].plot.bar(alpha=0.5)


plt.subplot(223)
age_range = pd.cut(train_data["Age"], np.arange(0, 90, 10))
train_data.groupby([age_range]).mean()['Survived'].plot.bar(alpha=0.5)


plt.subplot(224)
fare_range = pd.cut(train_data["Fare"], np.arange(0, 700, 100))
train_data.groupby([fare_range]).mean()['Survived'].plot.bar(alpha=0.5)

plt.subplots_adjust(hspace=0.34)
plt.subplots_adjust(bottom=0.17)
plt.subplots_adjust(wspace=0.28)
plt.savefig('ratio.png')

##### 预处理 ######
train_data = train_data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked','Fare'], axis=1)
# 填充 age
train_data['Age'] = train_data['Age'].fillna(value=train_data.Age.mean())
# 将性别转换为 0，1
train_data['Sex'] = preprocessing.LabelEncoder().fit_transform(train_data['Sex'])
X = train_data.drop(['Survived'], axis=1).values
y = train_data['Survived'].values

#### 生成决策树 #####
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.22)
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth = 3)
clf.fit (X_train, y_train)
name = train_data.drop(['Survived'], axis=1).columns
label = ["Unurvived","Survived"]
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=name, class_names=label,filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("tree.pdf")

#### 模型评估 #####
predict = clf.predict(X)
print(metrics.classification_report(y, predict))

scores = cross_validation.cross_val_score(clf, X, y, cv=10)
print scores.mean()





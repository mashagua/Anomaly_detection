# * coding:utf-8 *
#@author    :mashagua
#@time      :2019/7/25 19:47
#@File      :Univariate_Anomaly_Detection.py
#@Software  :PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.ensemble import IsolationForest
df=pd.read_excel('data/Sample - Superstore.xls')
print(df['Sales'].describe())
plt.scatter(range(df.shape[0]),np.sort(df['Sales'].values))
plt.xlabel('index')
plt.ylabel('Sales')
plt.title('Sales distribution')
sns.despine()
plt.show()
sns.distplot(df['Sales'])
plt.title('Distribution of Sales')
sns.despine()
plt.show()
print("Skewness: %f" % df['Sales'].skew())
print("Kurtosis: %f" % df['Sales'].kurt())
from sklearn.ensemble import IsolationForest
#建立100棵树
isolation_forest=IsolationForest(n_estimators=100)
#reshape（-1，1）的目的是变成1列向量，对数据进行拟合
isolation_forest.fit(df['Sales'].values.reshape(-1,1))
xx=np.linspace(df['Sales'].min(),df['Sales'].max(),len(df)).reshape(-1,1)
#XX=df['Sales'].values.reshape(-1,1)
anomaly_score=isolation_forest.decision_function(xx)
outlier=isolation_forest.predict(xx)
plt.figure(figsize=(10,4))
plt.plot(xx,anomaly_score,label='anomaly score')
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score),
                 where=outlier==-1, color='r',
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('Sales')
plt.show();


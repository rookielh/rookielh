import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB

datatrain = pd.read_csv("C:/Users/Master/Desktop/train.csv")
data_test = pd.read_csv("C:/Users/Master/Desktop/test.csv")
# print(datatrain.shape)    #(891, 12)
# print(datatrain.isnull().sum())  #查缺失

# 选取数据集特征，去掉几种无用特征
datatrain = datatrain.drop(labels=['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
# print(datatrain.head())

# 去除缺失值
datatrain = datatrain.dropna()
# print(datatrain)

# 属性转化为数值型
datatrain_dummy = pd.get_dummies(datatrain[['Sex']])
# print(type(datatrain[['Sex']]))

# 编码后和数据拼接
datatrain_conti = pd.DataFrame(datatrain,columns=['Survived','Pclass','Age'],index=datatrain.index)
# print(datatrain_conti)
datatrain = datatrain_conti.join(datatrain_dummy)
# print(datatrain)

X_train = datatrain.iloc[:,1:]
y_train = datatrain.iloc[:,0]
# print(X_train)
# print(y_train)

# 对test文件进行同样处理，去掉几种无用特征
datatest = data_test.drop(labels=['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
# print(datatest.head())
# print(datatest.isnull().sum())  # 查缺失

datatest = datatest.fillna(datatest.mean()['Age'])  # 填补缺失值
# print(datatest.isnull().sum())

# 属性转化为数值型
datatest_dummy = pd.get_dummies(datatest[['Sex']])

# 编码后和数据拼接
datatest_conti = pd.DataFrame(datatest,columns=['Pclass','Age'],index=datatest.index)
datatest = datatest_conti.join(datatest_dummy)
X_test = datatest

# 标准化
stdsc = StandardScaler()
X_train_conti_std = stdsc.fit_transform(X_train[['Age']])
X_test_conti_std = stdsc.fit_transform(X_test[['Age']])
# print(X_train_conti_std)
# print(X_test_conti_std)

# 将ndarray转为datatrainframe
X_train_conti_std = pd.DataFrame(data=X_train_conti_std,columns=['Age'],index=X_train.index)
X_test_conti_std = pd.DataFrame(data=X_test_conti_std,columns=['Age'],index=X_test.index)
# print(X_train_conti_std)
# print(X_test_conti_std)

# 有序分类变量Pclass
X_train_DF = X_train[['Pclass']]
X_test_DF = X_test[['Pclass']]
# print(X_train['Pclass'])
# print(X_train_cat)

# 无序已编码的分类变量
X_train_dummy = X_train[['Sex_female','Sex_male']]
X_test_dummy = X_test[['Sex_female','Sex_male']]
# print(X_train_dummy)

# 数据拼接
X_train_set = [X_train_DF,X_train_conti_std,X_train_dummy]
X_test_set = [X_test_DF,X_test_conti_std,X_test_dummy]
# print(X_train_set)
X_train = pd.concat(X_train_set,axis=1)
X_test = pd.concat(X_test_set,axis=1)
# print(X_train)
# print(X_test)

clf = BernoulliNB()
clf.fit(X_train,y_train)  # 用训练器数据拟合分类器模型
predicted = clf.predict(X_test)  # 对新数据预测
# print(predicted)

data_test['Survived'] = predicted.astype(int)  # 修改数据类型
data_test[['PassengerId','Survived']].to_csv('submission.csv',sep=',',index=0)  # 输出excel
# print(data_test)

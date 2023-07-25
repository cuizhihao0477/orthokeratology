# coding:utf-8

import pandas as pd
from fancyimpute import KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import joblib


def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    score = r2_score(y_true, y_predict)
    return score


def performance_metric2(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    y_true_arr = np.arr(y_true)
    y_predict_arr = np.arr(y_predict)

    # 数据集的平均值
    y_true_mean = np.mean(y_true_arr)
    ss_tot = 0
    ss_reg = 0
    ss_res = 0
    for index in range(y_true_arr):
        ss_tot += (y_true_arr[index] - y_true_mean) ** 2

    for index in range(y_true_arr):
        ss_reg += (y_predict_arr[index] - y_true_mean) ** 2

    for index in range(y_true_arr):
        ss_res += (y_predict_arr[index] - y_true_arr[index]) ** 2
    score = 1 - (ss_res / ss_tot)

    return score


def fit_randomforest_model(x, y):
    """ 基于输入数据 [X,y]，利于网格搜索找到最优的随机森林模型"""
    regressor = RandomForestRegressor()
    params = {"max_depth": np.arange(1, 20)}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc)
    # 基于输入数据 [X,y]，进行网格搜索
    grid = grid.fit(x, y)
    # 返回网格搜索后的最优模型
    return grid.best_estimator_
'''
需要更改的两个值
一个是加载的csv文件名称
需要预测的参数
'''
filepath = './crt_ordinary_test.csv'
need_predict = 'LZA' #需要预测的参数

data = pd.read_csv(filepath, encoding='GB2312') #载入不同的文件
data = data.dropna(axis=0,how='all') #去除都是nan的行
data = pd.DataFrame(KNN(k=6).fit_transform(data))


if 'toric' in filepath:
    toric = True
else:
    toric = False
# bool值，是否散光
data.columns = \
[['年龄', '球镜', 'Fk', 'Sk', '角膜直径', 'BC', 'RZD', 'LZA', '镜片直径', '基线眼轴', '半年眼轴', '1年眼轴'],
 ['年龄', '球镜', 'Fk', 'Sk', '角膜直径', 'BC', 'RZD1', 'RZD2', 'LZA1', 'LZA2', '镜片直径', '基线眼轴',
  '半年眼轴', '1年眼轴']][toric]  # fancyimpute填补缺失值时会自动删除列名
features = data.drop(labels=[['BC', 'RZD', 'LZA', '镜片直径', '基线眼轴', '半年眼轴', '1年眼轴'],
                             ['BC', 'RZD1', 'RZD2', 'LZA1', 'LZA2', '镜片直径', '基线眼轴', '半年眼轴', '1年眼轴']][
    toric],
                     axis=1)
prices = data[need_predict]

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

x_train = X_train.values.tolist()
x_test = X_test.values.tolist()

# 基于训练数据，获得最优模型
optimal_reg = fit_randomforest_model(X_train, y_train)
y_test_predict = optimal_reg.predict(X_test)
r2 = performance_metric(y_test, y_test_predict)
#模型的文件名称
if toric:
    joblib.dump(optimal_reg, 'model_crt_toric_' + prices.name + ".m")
else:
    joblib.dump(optimal_reg, 'model_crt_' + prices.name + ".m")



# 输出最优模型的预测特征值， 'max_depth' 参数，精确度R^2的值，还有预测csv文件含有的样本量
print("the predicting price is {:}.".format(need_predict))
print("Parameter 'max_depth' is {} for the optimal model.".format(optimal_reg.get_params()['max_depth']))
print("Optimal model has R^2 score {:,.2f} on test data".format(r2))
print("sample size is  {} on the data".format(data.shape[0] - 1))
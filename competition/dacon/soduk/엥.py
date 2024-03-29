# https://dacon.io/competitions/official/236230/mysubmission
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict, RandomizedSearchCV, GridSearchCV
import time
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
from scipy.stats import uniform, randint
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from xgboost import XGBRegressor
import tensorflow as tf
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import random
tf.random.set_seed(33)
np.random.seed(33)
random.seed(33)

warnings.filterwarnings("ignore")

#1. 데이터
csv_path = 'C:\\_data\\dacon\\soduk\\'

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

# def age_group(age):
#     if age < 30:
#         return 'Young'
#     elif age < 50:
#         return 'Middle-aged'
#     else:
#         return 'Elderly'

# train_csv['Income_Gains_Losses_Dividends'] = train_csv['Gains'] + train_csv['Losses'] + train_csv['Dividends']
# test_csv['Income_Gains_Losses_Dividends'] = test_csv['Gains'] + test_csv['Losses'] + test_csv['Dividends']

print(train_csv.shape)
print(test_csv.shape)

columns_to_drop2 = ['Gains','Losses', 'Dividends']
train_csv.drop(columns=columns_to_drop2, inplace=True)
test_csv.drop(columns=columns_to_drop2, inplace=True)


bins = [-1, 9, 19, 29, 39, 49, 59, 69, 79,89,99]  # 연령대 구간 정의
labels = [0, 1, 2, 3, 4, 5, 6, 7,8,9]  # 연령대 레이블

# bins = [-1, 19,29,  39, 49, 59, 69, 79,89,99]  # 연령대 구간 정의
# labels = [1, 2, 3, 4, 5, 6, 7,8,9]  # 연령대 레이블

# bins = [-1, 19, 39, 59, 69, 79,89,99]  # 연령대 구간 정의
# labels = [1, 2, 3, 4, 5, 6, 7,8,9]  # 연령대 레이블
train_csv['Age'] = pd.cut(train_csv['Age'], bins=bins, labels=labels)
test_csv['Age'] = pd.cut(test_csv['Age'], bins=bins, labels=labels)

print(train_csv["Age"])

# print(train_csv.shape)
# print(test_csv.shape)l

# 결과 확인
xy = train_csv
columns_to_drop = ['Income']
x = xy.drop(columns=columns_to_drop)
y = xy[columns_to_drop]


print(x.shape)
print(test_csv.shape)
# print(x["Total_Income"])
# lae = LabelEncoder()
# y = lae.fit_transform(y)
# print(x.head(10))


# x['Average_Income'] = round(x['Gains'] / x['Working_Week (Yearly)'] , 3)
# test_csv['Average_Income'] = round(test_csv['Gains'] / test_csv['Working_Week (Yearly)'], 3)

mode_value = test_csv['Household_Status'].mode()[0]
test_csv['Household_Status'].fillna(mode_value, inplace=True)

# print(x.dtypes)
# non_numeric_x = []
# for col in x.columns:
#     if x[col].dtype != 'int64 'and x[col].dtype != 'float64':
#         non_numeric_x.append(col)

for column in x.columns:
    if (x[column].dtype != 'int64'and x[column].dtype != 'float64'):
        encoder = LabelEncoder()
        x[column] = encoder.fit_transform(x[column])
        test_csv[column] = encoder.transform(test_csv[column])

x = x.astype('float32')
test_csv = test_csv.astype('float32')

random_state = 124


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=random_state)

# print(x_train.shape, y_train.shape)
# print(y_train.shape, y_test.shape)
# print(y_test)
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

# param_grid = {
#     'learning_rate': [0.01, 0.05, 0.001, 0.0001, 0.00001],
#     'max_depth': [4, 6, 8, 5, 9, 10],
#     # 'reg_lambda': [1, 3, 5],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'n_estimators': [100, 200, 300, 400],
#     'min_child_weight': [16, 32, 64, 128]  # min_child_samples -> min_child_weight로 변경
# }          

# model = GridSearchCV(XGBRegressor()
#                      , param_grid
#                      , cv=3
#                      , verbose=1
#                      , refit=True
#                      , n_jobs=-2
                    
#                      )
model = XGBRegressor(colsample_bytree=0.8, 
                    learning_rate=0.01, 
                    max_depth=11, 
                    min_child_weight=64,
                    n_estimators=400, 
                    subsample=0.8,
                    random_state=23
)




model.fit(x_train, y_train, eval_set=[(x_test, y_test)])

print("---------------------------------------------------------")
# print("최적의 파라미터 : ", model.best_params_)
# print('best_score : ', model.best_score_)
print("점수", model.score(x_test, y_test))
result = model.score(x_test, y_test)


# best_model = model.best_estimator_

# y_submit = best_model.predict(test_csv)
 #####################################################################################
y_submit = model.predict(test_csv)
# y_submit[y_submit < 0] = 0

submission_csv['Income'] = pd.DataFrame(y_submit.reshape(-1,1))
submission_csv.to_csv(csv_path + "오잉35.csv", index=False)

# 124 점수 0.3521863108186919

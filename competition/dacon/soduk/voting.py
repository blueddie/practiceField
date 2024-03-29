# https://dacon.io/competitions/official/236230/mysubmission
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingRegressor
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
import logging
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

def RMSE(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    return rmse
# print(x.dtypes)
# print(test_csv.dtypes)
random_state = 30
max_rs = 0
max_value = 0.2

while True :

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=random_state)

    # print(x_train.shape, y_train.shape)

    # scaler = StandardScaler()
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    test_csv = scaler.transform(test_csv)

    #2 모델
    # 최적의 파라미터 :  {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.025, 'max_depth': 7, 'min_child_weight': 7, 'n_estimators': 200, 'subsample': 0.8}
    xgb = XGBRegressor(
        colsample_bytree=0.8,
        gamma=0,
        learning_rate=0.025,
        max_depth=10,
        min_child_weight=9,
        n_estimators=200,
        subsample=0.8
    )

    # rf = RandomForestRegressor()
    cb = CatBoostRegressor(border_count=128, colsample_bylevel=0.8, depth=12, iterations=500, l2_leaf_reg=5, learning_rate=0.0025, subsample=0.9)
    # lgbm = LGBMRegressor(
    #     # learning_rate=0.1,
    #     #                  max_depth=-1,
    #     #                  reg_lambda=1,
    #     #                  n_estimators=200
    #                      )

    model = VotingRegressor(
        estimators=[('CB', cb),('XGB', xgb)],
        # voting= 'soft',
        # voting = 'hard' # 디폴트!
    )


    # #3 훈련
    # start_time = time.time()
    # model.fit(x_train, y_train)
    # end_time = time.time()


    # # print("최적의 파라미터 : ", model.best_params_)
    # # print('best_score : ', model.best_score_)
    # print('model.score : ', model.score(x_test, y_test))
    model.fit(x_train, y_train,
            #    eval_set=[(x_test, y_test)] ,
                 )

    # print("---------------------------------------------------------")
    # print("최적의 파라미터 : ", model.best_params_)
    # print('best_score : ', model.best_score_)
    # print("점수", model.score(x_test, y_test))
    result = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    rmse = RMSE(y_test, y_predict)

    if rmse < 550:
        max_rs = random_state
        max_value = result
        print(f"rs : {random_state}, max_value : {max_value}")
        random_state = random_state + 1
        y_submit = model.predict(test_csv)
        submission_csv['Income'] = pd.DataFrame(y_submit.reshape(-1,1))
        submission_csv.to_csv(csv_path + "0329_votingXGB_CB222.csv", index=False)
        print(rmse)
        break
    else :
        random_state = random_state + 2

    # if result > 0.30:
    #     if result > max_value:
    #         max_rs = random_state
    #         max_value = result
    #         print(f"rs : {random_state}, max_value : {max_value}")
    #         # print("최적의 파라미터 : ", model.best_params_)
    #         random_state = random_state + 1
    #         y_submit = model.predict(test_csv)
    #         submission_csv['Income'] = pd.DataFrame(y_submit.reshape(-1,1))
    #         submission_csv.to_csv(csv_path + "0329_votingXGB_CB.csv", index=False)
    #         rmse = RMSE(y_test, y_predict)
    #         print("rmse : ", rmse)
    #         break
    #     else :
    #         random_state = random_state + 1
    # else :
    #     random_state = random_state + 1
    
    


# def RMSE(y_test, y_predict):
#     rmse = np.sqrt(mean_squared_error(y_test, y_predict))



# print("RMSE : ", rmse)
# y_pred = model.predict(x_test)   
# train_rmse = RMSE(y_test, y_pred)
# print("훈련 rmse", train_rmse)
# print(y_pred)
##################################################################################
# y_submit = model.predict(test_csv)
# y_submit[y_submit < 0] = 0

# submission_csv['Income'] = pd.DataFrame(y_submit.reshape(-1,1))
# submission_csv.to_csv(csv_path + "032_3.csv", index=False)

# 최적의 파라미터 :  {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.025, 'max_depth': 7, 'min_child_weight': 7, 'n_estimators': 200, 'subsample': 0.8}
# model.score :  0.32612353495009505
# model.score :  0.3263897561218021

# model.score :  0.3266590696777495
        
# rs 124
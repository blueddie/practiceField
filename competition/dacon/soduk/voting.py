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

warnings.filterwarnings("ignore")

#1. 데이터
csv_path = 'C:\\_data\\dacon\\soduk\\'

train_csv = pd.read_csv(csv_path + "train.csv", index_col=0)
test_csv = pd.read_csv(csv_path + "test.csv", index_col=0)
submission_csv = pd.read_csv(csv_path + "sample_submission.csv")

# print(train_csv.shape)  #(20000, 22)
# print(test_csv.shape)   #(10000, 21)

# print(train_csv.describe())
#                 Age  Working_Week (Yearly)         Gains        Losses     Dividends        Income
# count  20000.000000           20000.000000  20000.000000  20000.000000  20000.000000  20000.000000
# mean      35.632500              34.943050    383.129500     40.202150    123.451450    554.565250
# std       17.994414              22.254592   4144.247487    279.182677   1206.949429    701.553155
# min        0.000000               0.000000      0.000000      0.000000      0.000000      0.000000
# 25%       23.000000               7.000000      0.000000      0.000000      0.000000      0.000000
# 50%       34.000000              52.000000      0.000000      0.000000      0.000000    500.000000
# 75%       47.000000              52.000000      0.000000      0.000000      0.000000    875.000000
# max       90.000000              52.000000  99999.000000   4356.000000  45000.000000   9999.000000

# print(pd.isna(train_csv).sum()) # 결측치 없음
# print(pd.isna(test_csv).sum()) # 결측치 Household_Status          1개 있음

# print(train_csv.columns)
# Index(['Age', 'Gender', 'Education_Status', 'Employment_Status',
#        'Working_Week (Yearly)', 'Industry_Status', 'Occupation_Status', 'Race',
#        'Hispanic_Origin', 'Martial_Status', 'Household_Status',
#        'Household_Summary', 'Citizenship', 'Birth_Country',
#        'Birth_Country (Father)', 'Birth_Country (Mother)', 'Tax_Status',
#        'Gains', 'Losses', 'Dividends', 'Income_Status', 'Income'],
#       dtype='object')

# print(test_csv.columns)
# Index(['Age', 'Gender', 'Education_Status', 'Employment_Status',
#        'Working_Week (Yearly)', 'Industry_Status', 'Occupation_Status', 'Race',
#        'Hispanic_Origin', 'Martial_Status', 'Household_Status',
#        'Household_Summary', 'Citizenship', 'Birth_Country',
#        'Birth_Country (Father)', 'Birth_Country (Mother)', 'Tax_Status',
#        'Gains', 'Losses', 'Dividends', 'Income_Status'],
#       dtype='object')

# target = Income

xy = train_csv.copy()
x_pred = test_csv.copy()

columns_to_drop = ['Income']
x = xy.drop(columns=columns_to_drop)
y = xy[columns_to_drop]

# print(x.shape, y.shape) # (20000, 21) (20000, 1)

# print(x.dtypes)
# Age                        int64
# Gender                    object
# Education_Status          object
# Employment_Status         object
# Working_Week (Yearly)      int64
# Industry_Status           object
# Occupation_Status         object
# Race                      object
# Hispanic_Origin           object
# Martial_Status            object
# Household_Status          object
# Household_Summary         object
# Citizenship               object
# Birth_Country             object
# Birth_Country (Father)    object
# Birth_Country (Mother)    object
# Tax_Status                object
# Gains                      int64
# Losses                     int64
# Dividends                  int64
# Income_Status             object
# print(pd.value_counts(x['Household_Status']))
mode_value = test_csv['Household_Status'].mode()[0]
test_csv['Household_Status'].fillna(mode_value, inplace=True)
# print(pd.isna(test_csv).sum()) # 결측치 최빈값으로 변경 최종적으로 없음
# print(pd.isna(x).sum()) # 훈련 데이터 결측치 없음 

# print(pd.value_counts(test_csv['Gender']))  # 불균형 없다고 판단
print(pd.value_counts(x['Education_Status']))
# print(pd.value_counts(x[]))
# print(pd.value_counts(x[]))
# print(pd.value_counts(x[]))

non_numeric_x = []
for col in x.columns:
    if x[col].dtype != 'int64':
        non_numeric_x.append(col)

# print(non_numeric_x)
# print("========")
# non_numeric_test = []
# for col in x.columns:
#     if test_csv[col].dtype != 'int64':
#         non_numeric_test.append(col)
# print(non_numeric_test)

# for col in non_numeric_x:
#     unique_values_train = set(x[col].unique())
#     unique_values_test = set(test_csv[col].unique())
#     common_values = unique_values_train.intersection(unique_values_test)
#     unique_values_only_df1 = unique_values_train - common_values
#     unique_values_only_df2 = unique_values_test - common_values
#     print("train :",unique_values_only_df1)
#     print("test :", unique_values_only_df2)
#     print("-------------------------------------------------------")

for column in x.columns:
    if (x[column].dtype != 'int64'):
        encoder = LabelEncoder()
        x[column] = encoder.fit_transform(x[column])
        test_csv[column] = encoder.transform(test_csv[column])

x = x.astype('float32')
test_csv = test_csv.astype('float32')

# print(x.dtypes)
# print(test_csv.dtypes)
random_state = 1
while True :

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    print(x_train.shape, y_train.shape)

    # scaler = StandardScaler()
    scaler = MinMaxScaler()
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

    if result > 0.3:
        if result > max_value:
            max_rs = random_state
            max_value = result
            print(f"rs : {random_state}, max_value : {max_value}")
            print("최적의 파라미터 : ", model.best_params_)
            random_state = random_state + 1
        else :
            random_state = random_state + 1
    else :
        random_state = random_state + 1
# def RMSE(y_test, y_predict):
#     rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    
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
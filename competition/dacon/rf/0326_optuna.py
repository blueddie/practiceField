import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import roc_auc_score
import optuna
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

# 1. 데이터
path = 'C:\\_data\\dacon\\rf\\'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

x = train_csv.drop(['login'], axis=1)
y = train_csv['login']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.27, random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 이전 시험의 최적 성능을 저장하기 위한 변수
best_auc = float('-inf')

def objective(trial):
    global best_auc
    
    # 하이퍼파라미터 탐색 공간 정의
    n_estimators = trial.suggest_int('n_estimators', 100, 700)
    max_depth = trial.suggest_int('max_depth', 4, 21)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 1.0)
    
    # 모델 생성
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features, 
        bootstrap=bootstrap,
        ccp_alpha=ccp_alpha,
        random_state=42
    )

    # 모델 학습
    model.fit(x_train, y_train)

    # 검증 세트에서의 AUC 계산
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 이전 시험보다 성능이 좋으면 최적 성능 및 파라미터 업데이트
    if auc > best_auc:
        best_auc = auc
    
    return auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5000)

# 최적의 하이퍼파라미터 및 AUC 출력
best_params = study.best_params
best_auc = study.best_value
print('Best parameters:', best_params)
print('Best AUC:', best_auc)

for param, value in best_params.items():
    if param in submission_csv.columns:
        submission_csv[param] = value

submission_csv.to_csv(path + "0401madoka2.csv", index=False)

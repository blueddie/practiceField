import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
from sklearn.metrics import roc_auc_score
import os
from sklearn.model_selection import train_test_split
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

data = pd.read_csv('C:\\_data\\dacon\\rf\\train.csv')

# person_id 컬럼 제거
x = data.drop(['person_id', 'login'], axis=1)
y = data['login']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, stratify=y)




# def objective(trial):
#     # 하이퍼파라미터 탐색 공간 정의
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 50, 200),
#         'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1),
#         'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
#         'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
#         'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
#         'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
#         'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
#         'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
#     }

#     model = RandomForestClassifier(**params)
#     score = cross_val_score(model, X_train, y_train, cv=5).mean()
def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        # 'min_samples_split': trial.suggest_float('min_samples_split', 0.0, 1.0),
        # 'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    # None 값 체크 및 제거
    # params = {key: value for key, value in params.items() if value is not None}

    clf = RandomForestClassifier(**params)
    clf.fit(x_train, y_train)
    # score = cross_val_score(model, X_train, y_train, cv=3).mean()
    return roc_auc_score(y_test, clf.predict_proba(x_test)[:,1])  # 최적화 대상인 목적 함수는 교차 검증 점수입니다.



# 최적화 스터디 생성
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

best_params = trial.params

# 제출 양식 불러오기
submit = pd.read_csv('C:\\_data\\dacon\\rf\\sample_submission.csv')

# 최적의 하이퍼파라미터를 제출 양식에 적용하여 예측 수행
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

# 예측 결과를 CSV 파일로 저장
submit.to_csv('C:\\_data\\dacon\\rf\\optuna33.csv', index=False)


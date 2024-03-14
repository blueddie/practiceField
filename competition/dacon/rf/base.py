import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

data = pd.read_csv('C:\\_data\\dacon\\rf\\train.csv')

# person_id 컬럼 제거
X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']

# GridSearchCV를 위한 하이퍼파라미터 설정
# param_search_space = {
#     'n_estimators': [100, 200, 300, 400],        # 트리의 개수
#     'max_depth': [10, 30, 50, 20, 12, 11],            # 트리의 최대 깊이
#     'min_samples_split': [2, 5, 10, 20, 15],        # 내부 노드를 분할하기 위한 최소한의 샘플 수
#     'min_samples_leaf': [1, 2, 4, 8,3 ,5],           # 리프 노드에 필요한 최소한의 샘플 수
#     'max_features': ['sqrt', 'log2'],   # 각 노드에서 사용할 최대 feature 수
#     'bootstrap': [True, False],                 # 부트스트랩 샘플 사용 여부
#     # 'criterion': ['mse', 'mae']                 # 분할 기준
# }

param_search_space = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, 12, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'min_weight_fraction_leaf': [0.0, 0.1],
    # 'max_features': ['auto'],
    'max_leaf_nodes': [None, 10, 20],
    'min_impurity_decrease': [0.0, 0.1],
    'bootstrap': [True]
}

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=rf, param_grid=param_search_space, cv=3, n_jobs=-2, verbose=2, scoring='roc_auc')
# random_search = RandomizedSearchCV(
#     estimator=rf,  # 모델
#     param_distributions=param_search_space,  # 하이퍼파라미터 공간
#     n_iter=100,  # 랜덤 샘플링 횟수
#     cv=3,  # 교차 검증 폴드 수
#     n_jobs=-2,  # 병렬 처리를 위한 작업 수
#     verbose=2,  # 로그 출력 레벨
#     scoring='roc_auc',  # 평가 지표
#     # random_state=13
# )
# # GridSearchCV를 사용한 학습
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_params, best_score

submit = pd.read_csv('C:\\_data\\dacon\\rf\\sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('C:\\_data\\dacon\\rf\\submitTest0313_grid.csv', index=False)
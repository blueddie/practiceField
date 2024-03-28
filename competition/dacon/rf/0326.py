import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from h2o.estimators import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch
import random
import os
import pdb
import h2o
h2o.init()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

# 1. 데이터
data = pd.read_csv('C:\\_data\\dacon\\rf\\train.csv')

# person_id 컬럼 제거
x = data.columns[:-1]
y = "login"

from h2o.frame import H2OFrame
data = H2OFrame(data)

# 2. 모델
rf = H2ORandomForestEstimator(seed=42)

hyper_parameters = {
    "ntrees": [50, 100, 150],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "min_weight_fraction_leaf": [0.0, 0.1, 0.2],
                    "max_leaf_nodes": [10, 20, 30],
                    "min_impurity_decrease": [0.0, 0.1, 0.2],
                    "bootstrap": [True, False]
                    }

# GridSearch 수행
grid_search = H2OGridSearch(rf, hyper_parameters)
grid_search.train(x=x, y=y, training_frame=data)

# 최적의 모델과 파라미터 출력
best_model = grid_search.models[0]
best_parameters = best_model.actual_params
print("Best Model:", best_model.model_id)
print("Best Parameters:", best_parameters)

submit = pd.read_csv('C:\\_data\\dacon\\rf\\sample_submission.csv')

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_parameters.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('C:\\_data\\dacon\\rf\\0326_h20_22.csv', index=False)

# Best Parameters: {'model_id': 'Grid_DRF_Key_Frame__upload_aec285534cb750e460f96350f224ef5.hex_model_python_1711433633090_1_model_1', 'training_frame': 'Key_Frame__upload_aec285534cb750e460f96350f224ef5.hex', 'validation_frame': None, 'nfolds': 
# 0, 'keep_cross_validation_models': True, 'keep_cross_validation_predictions': False, 'keep_cross_validation_fold_assignment': False, 'score_each_iteration': False, 'score_tree_interval': 0, 'fold_assignment': None, 'fold_column': None, 'response_column': 'login', 'ignored_columns': None, 'ignore_const_cols': True, 'offset_column': None, 'weights_column': None, 'balance_classes': False, 'class_sampling_factors': None, 'max_after_balance_size': 5.0, 'max_confusion_matrix_size': 20, 'ntrees': 50, 'max_depth': 10, 'min_rows': 1.0, 'nbins': 20, 'nbins_top_level': 1024, 'nbins_cats': 1024, 'r2_stopping': 1.7976931348623157e+308, 'stopping_rounds': 0, 'stopping_metric': None, 'stopping_tolerance': 0.001, 'max_runtime_secs': 0.0, 
# 'seed': 42, 'build_tree_one_node': False, 'mtries': -1, 'sample_rate': 0.632, 'sample_rate_per_class': None, 'binomial_double_trees': False, 'checkpoint': None, 'col_sample_rate_change_per_level': 1.0, 'col_sample_rate_per_tree': 1.0, 'min_split_improvement': 1e-05, 'histogram_type': 'UniformAdaptive', 'categorical_encoding': 'Enum', 'calibrate_model': False, 'calibration_frame': None, 'calibration_method': 'PlattScaling', 'distribution': 'gaussian', 'custom_metric_func': None, 'export_checkpoints_dir': None, 'check_constant_response': True, 'gainslift_bins': -1, 'auc_type': 'AUTO'
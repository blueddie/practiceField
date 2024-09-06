from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# RandomForest를 사용하여 RFE 적용
model = RandomForestClassifier()
selector = RFE(model, n_features_to_select=2)
X_new = selector.fit_transform(X, y)

# 선택된 특징 인덱스
selected_features = selector.get_support(indices=True)
print(f"선택된 특징: {selected_features}")

# RFE가 특징을 선택한 이유를 알기 위해 RandomForest에서의 중요도를 시각화
model.fit(X, y)  # RandomForest 모델을 학습
importances = model.feature_importances_

# 시각화
feature_names = iris.feature_names
plt.barh(np.arange(len(importances)), importances, align='center')
plt.yticks(np.arange(len(importances)), feature_names)
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Random Forest')

# 선택된 특징을 강조
for i in selected_features:
    plt.gca().get_yticklabels()[i].set_color("red")

plt.show()

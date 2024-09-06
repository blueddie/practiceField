from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# chi2를 사용하여 상위 2개의 중요한 특징 선택
selector = SelectKBest(score_func=chi2, k=2)
X_new = selector.fit_transform(X, y)

# 특징별 chi2 점수와 p-value
scores = selector.scores_

# 시각화
feature_names = iris.feature_names
print(feature_names)

"""
plt.barh(np.arange(len(scores)), scores, align='center')
plt.yticks(np.arange(len(scores)), feature_names)
plt.xlabel('Chi2 Score')
plt.title('Feature Importance Based on Chi2')

plt.show()

# 선택된 특징 출력
print(f"선택된 특징: {selector.get_support(indices=True)}")

"""
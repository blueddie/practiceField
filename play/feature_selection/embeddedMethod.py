from sklearn.datasets import load_iris
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target


# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso를 사용하여 임베디드 방식의 특징 선택
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# Lasso에 의해 선택된 중요한 특징들 출력
importance = lasso.coef_

# 시각화
plt.barh(iris.feature_names, importance)
plt.xlabel('Feature Importance (Lasso Coefficients)')
plt.title('Feature Importance with Lasso')
plt.show()

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# CSV 파일 로드
data = pd.read_csv('C:\\_data\\dacon\\birdClassification\\open\\train.csv')
test_csv = pd.read_csv('C:\\_data\\dacon\\birdClassification\\open\\test.csv')

print(data.shape)   # (15834, 3)
print(test_csv.shape)   # (6786, 2)

print(data.columns)

# 데이터셋 생성
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=data,
    x_col="img_path",
    y_col="label",
    target_size=(224, 224),  # 이미지 크기 조정
    batch_size=32,
    class_mode='categorical'  # 다중 클래스 분류
)


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
SEED = 42
IMAGE_SIZE = (224,224)
FILTERS = 16
np.random.seed(SEED)
tf.random.set_seed(SEED)

data_path = "C:\_data\dacon\\birdClassification\\" 
train_df = pd.read_csv(data_path + "train.csv")
test_df = pd.read_csv(data_path + "test.csv")

lbe = LabelEncoder()
lbe.fit_transform(train_df['label'])
train_df, val_df = train_test_split(train_df, test_size=0.3, stratify=train_df['label'], random_state=SEED)

# Data augmentation and data loaders setup
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='img_path',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='img_path',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,  # 이미지 경로가 절대 경로인 경우 None으로 설정
    x_col='img_path',  # 이미지 경로 열 이름
    y_col=None,  # 테스트 데이터에는 레이블이 없습니다
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # 레이블이 없으므로 None으로 설정
    shuffle=False  # 테스트 데이터 순서를 유지해야 합니다
)

from keras import backend as K
def F1Score(y_true, y_pred):
    # Precision 계산
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    
    # Recall 계산
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    
    # F1 Score 계산
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def build_model(num_classes):
    inputs = keras.Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

num_classes = num_classes = len(train_generator.class_indices)
model = build_model(num_classes)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', F1Score])

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=1,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

predictions = model.predict(test_generator, steps=len(test_generator))
predictions = np.argmax(predictions, axis=1)
submission_df = pd.read_csv(data_path + "sample_submission.csv")
submission_df['label'] = lbe.inverse_transform(predictions)

import time
timestr = time.strftime("%Y%m%d%H%M%S")
save_name = timestr
submission_df.to_csv(f'sample_submission_{save_name}.csv',index=False)
# %load C:\Users\AIA\Desktop\tf_cert_202306\starter02_cifar10_문제.py
# Question
#
# Create a classifier for the CIFAR10 dataset
# Note that the test will expect it to classify 10 classes and that the input shape should be
# the native CIFAR size which is 32x32 pixels with 3 bytes color depth

import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dropout, Dense
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder

def solution_model():
    cifar = tf.keras.datasets.cifar10
    
    # YOUR CODE HERE
    # 1. 데이터
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
#     print(x_train.shape, y_train.shape)
#     print(x_test.shape, y_test.shape)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    ohe = OneHotEncoder(sparse=False)
    y_train = ohe.fit_transform(y_train)
    y_test = ohe.transform(y_test)
    
#     print(len(y_train[0]))
    # 2. 모델
    model = Sequential()
    model.add(Conv2D(97, (3,3), activation='swish', input_shape=(32,32,3)))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (3,3), activation='swish'))
    model.add(Flatten())
#     model.add(Dense(16))
    model.add(Dense(10, activation='softmax'))
    
    # 3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(x_train, y_train, validation_split=0.2, batch_size=64, epochs=10)
    results = model.evaluate(x_test, y_test)
    print("정확도 : ", results[1])
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    # model.save("mymodel.h5")
# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create and train a classifier for the MNIST dataset.
# Note that the test will expect it to classify 10 classes and that the 
# input shape should be the native size of the MNIST dataset which is 
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Reshape, Input, Flatten
import numpy as np
def solution_model():
    mnist = tf.keras.datasets.mnist
    
    # YOUR CODE HERE
    (x_train,y_train), (x_test, y_test) = mnist.load_data()
    # print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
    # print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print(len(np.unique(y_train)))
    model = Sequential()
    model.add(Input(shape=(28,28)))
    model.add(Reshape((28,28,1)))
    model.add(Conv2D(32, kernel_size=(1,1), padding='same'))
    model.add(Conv2D(16, (3,3)))
    model.add(Conv2D(16, (3,3)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    # model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(x_train, y_train, batch_size=8, validation_data=(x_test, y_test), epochs=100)
    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.

if __name__ == '__main__':
    model = solution_model()
    # model.save("mymodel.h5")

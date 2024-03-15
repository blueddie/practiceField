# %load C:\Users\AIA\Desktop\tf_cert_202306\starter03_horse_sigmoid_문제.py
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
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly

import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dropout, Flatten, Dense
import numpy as np

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    local_zip = 'testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/testdata/')
    zip_ref.close()

    path_train = 'tmp/horse-or-human/'
    path_test = 'tmp/testdata/'
    
    train_datagen = ImageDataGenerator(
       
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting.)
         rescale= 1./255
        
    )
        
    validation_datagen = ImageDataGenerator(#Your Code here)
        rescale = 1./255
        
    )
    train_generator = train_datagen.flow_from_directory(
  
        #Your Code Here)
        path_train,
        target_size=(300,300),
        batch_size=32,
        class_mode='binary',
        color_mode='rgb',
        shuffle='True'
        
    )
    validation_generator = validation_datagen.flow_from_directory(
        #Your Code Here)
        path_test,
        target_size=(300,300),
        batch_size=32,
        class_mode='binary',
        color_mode='rgb',
    
    )
    
    x_train = []
    y_train = []
    
    for i in range(len(train_generator)):
        images, labels = train_generator.next()
        x_train.append(images)
        y_train.append(labels)
    
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    x_test = []
    y_test = []
    
    for i in range(len(validation_generator)):
        images, labels = validation_generator.next()
        x_test.append(images)
        y_test.append(labels)
    
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    
    
    model = tf.keras.models.Sequential([
        # Note the input shape specified on your first layer must be (300,300,3)
        # Your Code here
#         tf.keras.layers.Conv2D((16, (3,3), input_shape=(300, 300, 3), activation='relu')),
#         tf.keras.layers.Dropout(0.1),
#         tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Conv2D(16, (3,3), input_shape=(300, 300, 3), activation='relu'),  # 오타 수정
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),  # 'Con2D' 대신 'Conv2D' 사용 및 오타 수정
        tf.keras.layers.Flatten(),
        # This is the last layer. You should not change this code.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=1, verbose=1, epochs=2, validation_data=(x_test,y_test))
    
    
    return model

    
#     model.compile(#Your Code Here#)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#     model.fit(#Your Code Here#)

    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
    
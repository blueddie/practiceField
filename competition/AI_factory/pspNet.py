from keras.applications import ResNet101
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, Concatenate, Activation, BatchNormalization, Input
from keras.layers import AveragePooling2D
import os
import warnings
warnings.filterwarnings("ignore")
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
import sys
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import threading
import random
import rasterio
import os
import numpy as np
import sys
from sklearn.utils import shuffle as shuffle_lists
from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
np.random.seed(777)
tf.random.set_seed(777)
random.seed(777)
MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

class threadsafe_iter:
    """
    데이터 불러올떼, 호출 직렬화
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img

def get_img_765bands(path):
    img = rasterio.open(path).read((7,6,5)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img
    
def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg



@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, image_mode='10bands'):
   
    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr

    if image_mode == '762':
        fopen_image = get_img_762bands

    i = 0 
    # 데이터 shuffle
    while True:
        
        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
                i += 1 


        for img_path, mask_path in zip(images_path, masks_path):
            
            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []


def pyramid_pooling_module(input_tensor, bin_sizes):
    concat_list = [input_tensor]
    for bin_size in bin_sizes:
        x = AveragePooling2D(pool_size=(input_tensor.shape[1]//bin_size, input_tensor.shape[2]//bin_size))(input_tensor)
        x = Conv2D(256, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(input_tensor.shape[1]//bin_size, input_tensor.shape[2]//bin_size), interpolation='bilinear')(x)
        concat_list.append(x)
    return Concatenate()(concat_list)

def build_pspnet(input_shape, num_classes):
    base_model = ResNet101(include_top=False, weights=None, input_shape=input_shape)
    
    x = base_model.output
    x = pyramid_pooling_module(x, [1, 2, 3, 6])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)
    x = UpSampling2D(size=(input_shape[0]//x.shape[1], input_shape[1]//x.shape[2]), interpolation='bilinear')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model




# PSPNet 모델 빌드
# input_shape = (256, 256, 3)
# num_classes = 1  # 예시로 21개 클래스를 가정합니다.
# pspnet_model = build_pspnet(input_shape, num_classes)

# 모델 요약 확인
# pspnet_model.summary()


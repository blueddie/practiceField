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
from keras.callbacks import ReduceLROnPlateau
import segmentation_models as sm
import tensorflow_addons as tfa
tf.random.set_seed(33)
np.random.seed(33)
random.seed(33)
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

def get_img_762bands(path):
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
                


############################## 모델 정의 #################################


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='swish', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x

def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def conv2d_batchnorm(x, filters, kernel_size=(2, 2), activation='swish', padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == 'swish':
        x = Activation('swish')(x)
    return x

def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='swish', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='swish', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='swish', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('swish')(out)
    out = BatchNormalization(axis=3)(out)

    return out

def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='swish', padding='same')

    out = add([shortcut, out])
    out = Activation('swish')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='swish', padding='same')

        out = add([shortcut, out])
        out = Activation('swish')(out)
        out = BatchNormalization(axis=3)(out)

    return out

def MultiResUnet(height, width, n_channels):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    up6 = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6)

    up7 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32*4, up7)

    up8 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32*2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])

    return model

# def get_model(model_name, nClasses=1, input_height=256, input_width=256, n_filters=16, dropout=0.1, batchnorm=True, n_channels=3):
#     if model_name == 'MultiResUNet':
#         # 수정된 부분: MultiResUnet 함수 호출 시 필요한 인자들을 전달
#         return MultiResUnet(height=input_height, width=input_width, n_channels=n_channels)
#     else:
#         print("Model name not recognized. Please check the model name and try again.")
#         return None


# 두 샘플 간의 유사성 metric
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

# 픽셀 정확도를 계산 metric
def pixel_accuracy (y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)
 
    if (sum_t == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy    


# 사용할 데이터의 meta정보 가져오기

train_meta = pd.read_csv('C:\\_data\\dataset\\train_meta.csv')
test_meta = pd.read_csv('C:\\_data\\dataset\\test_meta.csv')


# 저장 이름
save_name = '0324final_line'

N_FILTERS = 18 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 1000 # 훈련 epoch 지정
BATCH_SIZE = 24 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'MultiResUNet' # 모델 이름
RANDOM_STATE = 14576 # seed 고정 42 -> 1113
# RANDOM_STATE = 25138 # seed 고정 42 -> 1113
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'C:\\_data\\dataset\\train_img\\'
MASKS_PATH = 'C:\\_data\\dataset\\train_mask\\'

# 가중치 저장 위치
OUTPUT_DIR = 'C:\\_data\\dataset\\output2\\'
WORKERS = 22

# 조기종료
EARLY_STOP_PATIENCE = 40

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}0326추가학습55.hdf5'.format(MODEL_NAME, save_name)
 
# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights0326추가학습55.h5'.format(MODEL_NAME, save_name)

# 사용할 GPU 이름
CUDA_DEVICE = 0


# 저장 폴더 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

try:
    np.random.bit_generator = np.random._bit_generator
except:
    pass


# train : val = 8 : 2 나누기
x_tr, x_val = train_test_split(train_meta, test_size=0.15, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val))

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")


#miou metric
Threshold  = 0.25
def miou(y_true, y_pred, smooth=1e-6):
    # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > Threshold , tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # mIoU 계산
    iou = (intersection + smooth) / (union + smooth)
    miou = tf.reduce_mean(iou)
    return miou

# model 불러오기
learning_rate = 0.000008
model = MultiResUnet(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1], n_channels=N_CHANNELS,)
# optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)  # 1e-4 = 0.0001
optimizer = Adam(learning_rate=learning_rate)  # 1e-4 = 0.0001

model.compile(
              optimizer=optimizer,
            #   loss = sm.losses.binary_focal_dice_loss,
              loss=sm.losses.bce_jaccard_loss, 
              metrics=['accuracy', sm.metrics.iou_score, miou])
model.summary()
# model.load_weights('C:\\_data\\dataset\\output\\checkpoint-MultiResUNet-0324final_line-epoch_010324Final44111.hdf5')
model.load_weights('C:\\_data\\dataset\\output2\\checkpoint-MultiResUNet-0324final_line-epoch_020326추가학습3.hdf5')

# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='val_loss', verbose=1,
save_best_only=True, mode='min')
# Reduce
rlr = ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.8, mode='min')


# from keras.callbacks import LearningRateScheduler
# import tensorflow as tf
# import numpy as np
# def cosine_decay(epoch, max_epochs=EPOCHS, initial_learning_rate=learning_rate, min_learning_rate=0.00001):
#     alpha = min_learning_rate / initial_learning_rate
#     cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * epoch / max_epochs))
#     decayed = (1 - alpha) * cosine_decay + alpha
#     decayed_learning_rate = initial_learning_rate * decayed
#     return decayed_learning_rate
# rlr = LearningRateScheduler(lambda epoch: cosine_decay(epoch, max_epochs=EPOCHS, initial_learning_rate=learning_rate), verbose=1)



print('---model 훈련 시작---')
history = model.fit(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es, rlr],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH
)
print('---model 훈련 종료---')

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))


# model.load_weights('C:\\_data\\dataset\\output\\checkpoint-MultiResUNet-0324final_line-epoch_010326검증.hdf5')

# y_pred_dict = {}

# for i in test_meta['test_img']:
#     img = get_img_762bands(f'C:\\_data\\dataset\\test_img\\{i}')
#     y_pred = model.predict(np.array([img]), batch_size=1, verbose=0)
    
#     y_pred = np.where(y_pred[0, :, :, 0] > 0.26, 1, 0) # 임계값 처리
#     y_pred = y_pred.astype(np.uint8)
#     y_pred_dict[i] = y_pred

# from datetime import datetime
# dt = datetime.now()
# joblib.dump(y_pred_dict, f'C:\\_data\\dataset\\output\\검증.pkl')
# print(f'끝. : y_pred_{dt.day}_{dt.hour}_{dt.minute}.pkl ')

# 67 체크포인트 임계값 26

# model.load_weights('C:\\_data\\dataset\\output\\checkpoint-MultiResUNet-0324final_line-epoch_1040324Final44.hdf5') 로드 후 0.00001 재훈련

# c:\_data\dataset\output\checkpoint-MultiResUNet-0324final_line-epoch_010324Final44111.hdf5 프레딕 임계값 0.26 
# 가상환경 bull







# checkpoint-MultiResUNet-0324final_line-epoch_020324Final44111.hdf5 로드 후 0.000005 로 재훈련
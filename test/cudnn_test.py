import tensorflow as tf

# TensorFlow에서 CuDNN이 사용되는지 여부 확인
print("CuDNN이 사용 중인지:", tf.test.is_built_with_cuda())

import torch

# PyTorch에서 CuDNN이 사용되는지 여부 확인
print("CuDNN이 사용 중인지:", torch.backends.cudnn.enabled)

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])

import ctypes

def get_cudnn_version():
    libcudnn = ctypes.cdll.LoadLibrary("cudnn64_8.dll")
    version = ctypes.c_int()
    libcudnn.cudnnGetVersion(ctypes.byref(version))
    return version.value

cudnn_version = get_cudnn_version()
print("cuDNN version:", cudnn_version)
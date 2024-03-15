import tensorflow as tf

# TensorFlow에서 CuDNN이 사용되는지 여부 확인
print("CuDNN이 사용 중인지:", tf.test.is_built_with_cuda())

import torch

# PyTorch에서 CuDNN이 사용되는지 여부 확인
print("CuDNN이 사용 중인지:", torch.backends.cudnn.enabled)
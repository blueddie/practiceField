import tensorflow as tf

if tf.test.is_gpu_available():
    print("CUDA 사용 가능")
else:
    print("CUDA 사용 불가능")

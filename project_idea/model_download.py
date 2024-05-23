import os
import tarfile
import wget

# Download and extract model
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

DEST_DIR = MODEL_NAME
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

wget.download(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

with tarfile.open(MODEL_FILE) as tar:
    tar.extractall(path=DEST_DIR)

print("Model downloaded and extracted.")

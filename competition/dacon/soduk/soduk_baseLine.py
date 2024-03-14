import numpy as np
import random
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(77) # Seed 고정

import pandas as pd

train = pd.read_csv('C:\\_data\\dacon\\soduk\\train.csv')
test = pd.read_csv('C:\\_data\\dacon\\soduk\\test.csv')

# display(train.head(3))
# display(test.head(3))
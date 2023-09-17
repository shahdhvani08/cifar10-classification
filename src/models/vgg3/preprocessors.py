import numpy as np
import cv2
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
# fetch the root directory to read config
import os, sys
config_dir = (os.path.abspath(Path(__file__).parent.parent.parent.parent) + '/configs/')
sys.path.append(config_dir)
import config
import data_management as dm

class TargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, encoder = LabelEncoder()):
        self.encoder = encoder

    def fit(self, X, y=None):
        # note that x is the target in this case
        self.encoder.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X = to_categorical(self.encoder.transform(X.ravel()))
        return X
    
    
class CreateDataset(BaseEstimator, TransformerMixin):

    def __init__(self, image_size = 32):
        self.image_size = image_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # convert from integers to floats
        X = X.astype('float32')

        # normalize to range 0-1
        X = X / 255.0
  
        return X
    

if __name__ == '__main__':
    import data_management as dm
    trainX, trainY, testX, testY = dm.load_dataset()
    
    enc = TargetEncoder()
    enc.fit(trainY)
    y_train = enc.transform(trainY)
    
    dataCreator = CreateDataset()
    X_train = dataCreator.transform(trainX) 
 
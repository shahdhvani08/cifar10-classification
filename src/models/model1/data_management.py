import pandas as pd
from glob import glob
import os, sys


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from keras import datasets
from keras.utils import to_categorical
from keras.models import load_model
from scikeras.wrappers import KerasClassifier

import model as m
from pathlib import Path
# fetch the root directory to read config

config_dir = (os.path.abspath(Path(__file__).parent.parent.parent.parent) + '/configs/')
sys.path.append(config_dir)
print(config_dir)
import config

def load_dataset():
    # load cifar10 dataset
    print("Loading dataset")
    (trainX, trainy), (testX, testy)  = datasets.cifar10.load_data()
    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

    return trainX, trainy, testX, testy


def save_pipeline_keras(model):
    
    joblib.dump(model.named_steps['dataset'], config.PIPELINE_PATH)
    joblib.dump(model.named_steps['cnn_model'].classes_, config.CLASSES_PATH)
    joblib.dump(model.named_steps['cnn_model'].classes_, config.MODEL_PATH)
    #model.named_steps['cnn_model'].model.save(config.MODEL_PATH)
    
    
def load_pipeline_keras():
    dataset = joblib.load(config.PIPELINE_PATH)
    
    build_model = lambda: load_model(config.MODEL_PATH)
    
    classifier = KerasClassifier(build_fn=build_model,
                          batch_size=config.BATCH_SIZE, 
                          validation_split=10,
                          epochs=config.EPOCHS,
                          verbose=2,
                          callbacks=m.callbacks_list,
                          #image_size = config.IMAGE_SIZE
                          )
    
    classifier.classes_ = joblib.load(config.CLASSES_PATH)
    classifier.model = build_model()
    
    return Pipeline([
        ('dataset', dataset),
        ('cnn_model', classifier)
    ])
    
    
if __name__ == '__main__':
        
    trainX, trainY, testX, testY = load_dataset()
    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)
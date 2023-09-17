import joblib
import os, sys
import numpy as np
import data_management as dm
from pathlib import Path
# fetch the root directory to read config
from pathlib import Path
config_dir = (os.path.abspath(Path(__file__).parent.parent.parent.parent) + '/configs/')
sys.path.append(config_dir)
print(config_dir)
import config
import wandb

import pipeline as pipe
import preprocessors as pp


def run_training(save_result: bool = True):
    
    trainX, trainY, testX, testY = dm.load_dataset()
    
    enc = pp.TargetEncoder()
    print(trainY.shape)
    enc.fit(trainY)
    y_train = enc.transform(trainY)
    print(np.unique(y_train))

    history = pipe.pipe.fit(trainX, y_train)

    print(history)
    
    if save_result:
        joblib.dump(enc, config.ENCODER_PATH)
        dm.save_pipeline_keras(pipe.pipe)


if __name__ == '__main__':
    run_training(save_result=True)
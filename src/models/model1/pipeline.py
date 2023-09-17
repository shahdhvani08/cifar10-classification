from sklearn.pipeline import Pipeline
from pathlib import Path
import os, sys
config_dir = (os.path.abspath(Path(__file__).parent.parent.parent.parent) + '/configs/')
sys.path.append(config_dir)
print(config_dir)
import config
import preprocessors as pp
import model
import numpy as np

pipe = Pipeline([
                ('dataset', pp.CreateDataset()),
                ('cnn_model', model.cnn_clf)
            ])


if __name__ == '__main__':
    
    from sklearn.metrics import  accuracy_score
    import data_management as dm
    from pathlib import Path
    import os, sys
    config_dir = (os.path.abspath(Path(__file__).parent.parent.parent.parent) + '/configs/')
    sys.path.append(config_dir)
    print(config_dir)
    import config
    
    trainX, trainY, testX, testY = dm.load_dataset()
    
    enc = pp.TargetEncoder()
    print(trainY.shape)
    enc.fit(trainY)
    y_train = enc.transform(trainY)
    print(np.unique(y_train))

    history = pipe.fit(trainX, y_train)
    
    test_y = enc.transform(testY)
    predictions = pipe.predict(testX)
    
    acc = accuracy_score(enc.encoder.transform(testY),
                   predictions,
                   normalize=True,
                   sample_weight=None)
    
    print('Acuracy: ', acc)
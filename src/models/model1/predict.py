import data_management as dm
from pathlib import Path
# fetch the root directory to read config
from pathlib import Path
import os, sys
config_dir = (os.path.abspath(Path(__file__).parent.parent.parent.parent) + '/configs/')
sys.path.append(config_dir)
print(config_dir)
import config
import preprocessors as pp
import numpy as np
from sklearn.metrics import  accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

def make_prediction(*, path_to_images) -> float:
    """Make a prediction using the saved model pipeline."""
    
    trainX, trainY, testX, testY = dm.load_dataset()
    pipe = dm.load_pipeline_keras()
    predictions = pipe.predict(testX)
    #response = {'predictions': predictions, 'version': _version}

    return predictions


if __name__ == '__main__':
    
    import joblib

    trainX, trainY, testX, testY = dm.load_dataset()
    pipe = joblib.load(config.PIPELINE_PATH)
    
    testX = pipe.transform(testX)
    print(pipe)
    pipe = dm.load_pipeline_keras()
    print(testX)

    enc = pp.TargetEncoder()
    print(testY.shape)
    enc.fit(testY)
    test_y = enc.transform(testY)
    print(enc._get_param_names())
    print(test_y)
             
    print(pipe.named_steps['classes'])
    print(pipe.named_steps['cnn_model'])


    modeltest = pipe.named_steps['cnn_model']
    predictions = modeltest.predict(testX)
    print(predictions)
    print(predictions.shape)

    print(test_y)

    y_pred=np.argmax(predictions, axis=1)
    y_test=np.argmax(test_y, axis=1)

    confusion = confusion_matrix(y_test, y_pred)
    print(confusion)

    #importing accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

    
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['airplane',
	                                                'automobile',
                                                    'bird',
                                                    'cat',
                                                    'deer',
                                                    'dog',
                                                    'frog',
                                                    'horse',
                                                    'ship',
                                                    'truck']))

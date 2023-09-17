# DATA
DATA_FOLDER = ''

# MODEL FITTING
BATCH_SIZE = 48
EPOCHS = 200

# MODEL PARAMETERS (SHOULD BE UPDATED AFTER HYPERPARAMETER TUNING CONSIDERING THE TUNED PARAMETERS)
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128
dropout_conv = 0.2
dropout_dense = 0.2

# MODEL PERSISTING
MODEL_PATH = 'artifacts/cnn_model.h5'
PIPELINE_PATH = 'artifacts/cnn_pipe.pkl'
CLASSES_PATH = 'artifacts/classes.pkl'
ENCODER_PATH = 'artifacts/encoder.pkl'
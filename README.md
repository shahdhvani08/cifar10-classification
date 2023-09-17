# cifar10-classification

This code was implemented on Mac OS M2.

This file describes the steps to replicate the development envrionment and how to build this DL project.

Folder structure:

```
cifar10-classification
├── configs
│   └── __pycache__
├── logs
│   ├── fit
│   │   ├── 20230916-143017
│   │   │   ├── train
│   │   │   └── validation
│   │   ├── 20230916-143956
│   │   │   ├── train
│   │   │   └── validation
│   │   ├── 20230916-223934
│   │   │   ├── train
│   │   │   └── validation
│   │   └── 20230916-230036
│   │       ├── train
│   │       └── validation
│   └── hyperband
├── notebooks
├── reports
└── src
 └── models
     └── model1
         ├── __pycache__
         └── artifacts
```

# Environment Setup

1.  Download and install miniconda from the official Anaconda website: [https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)
2.  Use the following code to create the conda env using environment.yml:  
    `conda env create -f environment.yml`
3.  Test if all the required libraries are installed:  
    `import tensorflow as tf`  
    `import matplotlib.pyplot as plt`  
    `import numpy as np`  
    `import pandas as pd`  
    `import sklearn`

# Training

1.  Go the folder src/models/model1/
2.  RUN: `python train_pipeline.py`

# Predictions

1.  Go the folder src/models/model1/
2.  RUN: `python predict.py`

# Research Environment 

1.  CNN_research_env.ipynb - implementation of VGG (3 blocks) in TensorFlow and Hyperparameter tuning
2.  Depthwise_Convolutions_CNN.ipynb - Using depthwise convolutions to reduce model size

# Python Scripts

1.  data_management.py - Loads the cifar10 data, save and load Pipelines
2.  model.py - Defines the VGG (3 blocks) architecture
3.  pipeline.py - Complete pipeline from training and predictions
4.  predict.py - Load pipeline, un predictions, compute inference time per CPU thread and performance analysis
5.  preprocessors.py - Convert target to categorical and normalise data
6.  train_pipeline.py - Train the model and save the model in artifacts folder
7.  configs/config.py - Define model parameters

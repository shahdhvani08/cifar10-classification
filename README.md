# cifar10-classification
This code was implemented on Mac OS M2.

This file describes the steps to replicate the development envrionment and how to build this DL project.

1. Download and install miniconda from the official Anaconda website - https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html

2. Use the following code to create the conda env using environment.yml
conda env create -f environment.yml

3. Test if all the required libraries are installed:
 import tensorflow as tf
 import matplotlib.pyplot as plt
 import numpy as np
 import pandas as pd
 import sklearn


4. Folder and File structure:
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



__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

# Caution: Auto-PyTorch uses scikit-learn 0.20.2. However, AutoKeras uses different version of the one.
# Therefore, I recommend to create another vitrual enviroment for Auto-PyTourch.

from autoPyTorch import AutoNetClassification

# data and metric imports
import sklearn.metrics
import pandas as pd
import numpy as np
import cv2
import os
import torch

import Settings
import SettingsPrivate

dataList = pd.read_csv("trainForAutoPyTorch.csv", header=0)

X=[]
y=[]
for index, row in dataList.iterrows():
    #center cropping
    img=cv2.imread(os.path.join(SettingsPrivate.DATASET_DIR, row["File Name"]))
    w=img.shape[1]
    h=img.shape[0]
    edge=np.min(img.shape)
    img=img[(h-edge)//2:(h+edge)//2,(w-edge)//2:(w+edge)//2,:]
    img=cv2.resize(img,(224,224))
    X.append(img)

    y.append(row["Label"])
X=np.array(X,dtype=np.float64)/255.
y=np.array(y,dtype=np.int64)

# running Auto-PyTorch
autonet = AutoNetClassification("tiny_cs", budget_type='epochs', min_budget=1, max_budget=9, num_iterations=1, log_level='debug', use_pynisher=False)

res = autonet.fit(X_train=X, Y_train=y, cross_validator="k_fold", cross_validator_args={"n_splits": 5})

torch.save(autoPyTorch.state_dict(), Settings.MODEL_PYTORCH_PATH)

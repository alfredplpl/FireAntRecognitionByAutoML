__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

# This code don't works because auto-pytorch has a bag about target values.
# dataset_info.categorical_features = [dataset_info.categorical_features[i] for i, is_nan in enumerate(all_nan) if not is_nan]
# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

from autoPyTorch import AutoNetClassification

# data and metric imports
import sklearn.metrics
import pandas as pd
import numpy as np
import cv2
import os
import torch

import Settings

dataList = pd.read_csv("trainForAutoPyTorch.csv", header=0)

X=[]
y=[]
for index, row in dataList.iterrows():
    #center cropping
    img=cv2.imread(os.path.join(Settings.DATASET_DIR, row["File Name"]))
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
autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='info',
                                    max_runtime=60000,
                                    min_budget=1,
                                    max_budget=30000)

autoPyTorch.fit(X, y, validation_split=0.2)

torch.save(autoPyTorch.state_dict(), Settings.MODEL_PATH)

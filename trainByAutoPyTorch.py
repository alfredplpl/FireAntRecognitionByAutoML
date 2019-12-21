from autoPyTorch import AutoNetClassification

# data and metric imports
import sklearn.metrics
import pandas as pd
import numpy as np
import cv2
import os
import torch

import Settings

dataList = pd.read_csv("datalist.csv", header=0)

X=[]
y=[]
for index, row in dataList.iterrows():
    #center cropping
    img=cv2.imread(os.path.join(Settings.DATASET_DIR, row["file_name"]))
    w=img.shape[1]
    h=img.shape[0]
    edge=np.min(img.shape)
    img=img[(h-edge)//2:(h+edge)//2,(w-edge)//2:(w+edge)//2,:]
    img=cv2.resize(img,(224,224))
    X.append(img)

    y.append(row["category_id"])
X=np.array(X)
y=np.array(y)

# running Auto-PyTorch
autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='info',
                                    max_runtime=6000,
                                    min_budget=1000,
                                    max_budget=3000)

autoPyTorch.fit(X, y, validation_split=0.3)

torch.save(autoPyTorch.state_dict(), Settings.MODEL_PATH)

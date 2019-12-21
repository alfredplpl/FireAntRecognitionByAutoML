import autokeras as ak

# data and metric imports
import sklearn.metrics
import pandas as pd
import numpy as np
import cv2
import os
import torch

import Settings

dataList = pd.read_csv("trainForAutokeras.csv", header=0)

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

clf = ak.ImageClassifier(verbose=True, augment=True)
clf.fit(X, y)

clf.export_autokeras_model( Settings.MODEL_PATH)
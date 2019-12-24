__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

import autokeras as ak
from sklearn.model_selection import train_test_split

from keras.utils import plot_model
from keras.models import load_model
import numpy as np
import pandas as pd
import cv2
import os

import Settings
import SettingsPrivate
import time

TMP_MODEL_PATH=os.path.join(Settings.TMP_PATH,str(int(time.time())))
os.mkdir(TMP_MODEL_PATH)

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
    img=cv2.resize(img,(224,224 ))
    X.append(img)

    y.append(row["Label"])
X=np.array(X,dtype=np.float64)/255.
y=np.array(y,dtype=np.int64)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1)
clf = ak.ImageClassifier(path=TMP_MODEL_PATH,augment=True,verbose=True)
clf.fit(X_train, y_train,time_limit=10*60 \)
clf.final_fit(X_train, y_train, X_test, y_test, retrain=True)
results = clf.evaluate(X_test, y_test)
print(results)

clf.export_autokeras_model(Settings.MODEL_PATH)

# References
# https://qiita.com/hideki/items/24f6c776828ac918a109
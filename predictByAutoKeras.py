__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

#import sys
import cv2
import imageio
import numpy as np
from autokeras.utils import pickle_from_file

import Settings

if __name__ == "__main__":
    X=imageio.imread("images/fire_ant.jpg")
    X=X[np.newaxis,:,:,:]
    clf = pickle_from_file(Settings.MODEL_PATH)
    result = clf.predict(X)
    result = "Fire ant" if result[0]==1 else "Not fire ant"
    img=cv2.cvtColor(X[0],cv2.COLOR_BGR2RGB)
    cv2.imshow(f"Result:{result}. Input any key to exit.",img)
    cv2.waitKey(0)

#if len(sys.argv) == 1:
# References
# https://qiita.com/petitviolet/items/aad73a24f41315f78ee4
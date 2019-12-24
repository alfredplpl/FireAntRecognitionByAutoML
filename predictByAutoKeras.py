__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

import sys
import imageio
import numpy as np
from autokeras.utils import pickle_from_file

import Settings

if __name__ == "__main__":
    if len(arguments) == 1:
        return usage
        X=imageio.imread("images/fire_ant.jpg")
        X=X[np.newaxis,:,:,:]
        clf = pickle_from_file(Settings.MODEL_PATH)
        result = clf.predict(X)
        print(result)

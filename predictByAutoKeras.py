__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

import imageio
import numpy as np
from autokeras.utils import pickle_from_file

import Settings

X=imageio.imread(Settings.TEST_PATH)
X=X[np.newaxis,:,:,:]
clf = pickle_from_file(Settings.MODEL_PATH)
result = clf.predict(X)
print(result)

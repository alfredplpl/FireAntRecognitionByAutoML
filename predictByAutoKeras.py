__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

import imageio
from autokeras.utils import pickle_from_file

import Settings

X=imageio.imread(Settings.TEST_PATH)
clf = pickle_from_file("models/cllasifier")
result = clf.predict(X)
print(result)

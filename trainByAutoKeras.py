__author__ = "Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"

import autokeras as ak
from autokeras.image.image_supervised import load_image_dataset

import Settings

X, y = load_image_dataset(csv_file_path="trainForAutokeras.csv",images_path=Settings.DATASET_DIR)
clf = ak.ImageClassifier(path=Settings.TMP_PATH,augment=True,verbose=True)
clf.fit(X, y,time_limit=10*60)

clf.export_autokeras_model(Settings.MODEL_PATH)
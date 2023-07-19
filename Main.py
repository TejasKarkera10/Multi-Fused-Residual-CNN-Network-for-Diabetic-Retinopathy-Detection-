import zipfile
import pandas as pd
import os
import shutil
import Models
import torch
import Config
import os
import cv2
import Utils
import time
import pickle
import logging
import itertools
import numpy as np
import tensorflow as tf
import python_splitter
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import applications
from skimage import exposure
from numpy.core.fromnumeric import resize

from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam

from sklearn.utils import class_weight
from tensorflow.keras.models import load_model
from keras.utils.layer_utils import count_params
from sklearn.metrics import accuracy_score, mean_squared_error, mean_squared_log_error, classification_report, confusion_matrix, roc_curve, auc

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import AveragePooling2D, AlphaDropout, Activation, Add, BatchNormalization, Concatenate, Layer, ReLU, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout

#Models
from tensorflow.keras.applications.efficientnet import EfficientNetB0 as trainable_model_a
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as trainable_model_b
from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as trainable_model_c

#PREVENT ERROR UNCESSARY MESSAGES
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Packages successfully imported!")

PATH_GDRIVE_ZIP = "/content/drive/MyDrive/DBTRD/aptos2019-blindness-detection.zip"

with zipfile.ZipFile(PATH_GDRIVE_ZIP, 'r') as ZIP:
    ZIP.extractall("APTOS")

Data = pd.read_csv("/content/APTOS/train.csv")

os.mkdir("FinalData")
os.mkdir("FinalData/0")
os.mkdir("FinalData/1")
os.mkdir("FinalData/2")
os.mkdir("FinalData/3")
os.mkdir("FinalData/4")

for ix in range(len(Data['id_code'])):
  img = Data.iloc[ix , 0]
  lbl = Data.iloc[ix , 1]
  shutil.move(os.path.join("/content/APTOS/train_images" , img + '.png') , os.path.join("FinalData" , str(lbl) , img + '.png'))

python_splitter.split_from_folder("/content/FinalData", train=0.8, test=0.1, val=0.1)

train_generator, validation_generator, test_generator, nb_train_samples, nb_validation_samples = Utils.PREP_DATA()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.array([0,1,2,3,4]),y=Data['diagnosis'].values)
class_weights = torch.tensor(class_weights,dtype=torch.float).to(DEVICE)
print(class_weights)


FUCNN_MODEL = Models.mfurecnn_builder(Models.models, Config.MODEL_INPUT)
FUCNN_MODEL._name = "MFuReCNN"
model_name = FUCNN_MODEL._name

print()
print()
print("PLEASE CHECK THE MODEL UP TO THE END")
print()
print()
print()

FUCNN_MODEL.summary()
print("The", model_name, "is now complete and ready for compilation and training!")

if not os.path.exists(Config.MODEL_DIR):
  os.mkdir(Config.MODEL_DIR)
  print("MODEL DIRECTORY MADE")
else:
  print("MODEL DIRECTORY ALREADY EXISTS")

print()
print('-'*70)
print('Model directory is available for saving the', model_name, 'model!')
print('-'*70)

FUCNN_MODEL.compile(optimizer = Config.OPTIMIZER, loss = 'categorical_crossentropy', metrics = ["accuracy"])
reduceLr =  ReduceLROnPlateau(monitor = 'val_acc', factor = 0.5, patience = 2, verbose = 1, mode = 'max')

callbacks = [reduceLr]

print('-'*50)
print('Successfully compiled the', model_name, 'model!')
print('You may now proceed in training the', model_name, 'model!')
print('-'*50)

#Set training time
start_time = time.time()
print('*'*50)
print("Training", model_name)
print('*'*50)
print('-'*50)
print("Training time is being measured")
print('-'*50)

class_weights = {i : class_weights[i] for i in range(5)}
history = FUCNN_MODEL.fit(train_generator, steps_per_epoch = nb_train_samples // Config.BATCH_SIZE, epochs= Config.EPOCHS, validation_data = validation_generator, callbacks = callbacks, validation_steps = nb_validation_samples // Config.BATCH_SIZE, verbose = 1,class_weight = class_weights)

print()
print("MODEL SERIALIZING WAIT FOR A MOMENT...")
elapsed_time = time.time() - start_time
train_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print()
print()
print(train_time, 'train_time')
print()
print(elapsed_time, 'Seconds')
print()
print()
print("MODEL SERIALIZATION DONE!")
# EfficientNetB0

import Config
from tensorflow.keras import Model, layers
import tensorflow as tf

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
build_A = Config.DCNN_A + "_builder"

def Builder_A(model_input):
  builder_a = trainable_model_a(weights='imagenet',
                                include_top=False,
                                input_tensor=model_input)

  for layer in builder_a.layers:
    layer.trainable = False

  for layer in builder_a.layers:
    layer._name = layer.name + '_' + Config.DCNN_A

  for batchnorm in builder_a.layers:
    batchnorm.trainable  = False

# Getting one core block output
  x = builder_a.layers[-17].output

  x = Conv2D(192, 1, padding='valid', activation='selu',  kernel_initializer='lecun_normal')(x)
  x = AveragePooling2D(1,1)(x)
  x = AlphaDropout(0.2)(x)

  dcnn_a = Model(inputs=builder_a.input, outputs=x, name=Config.DCNN_A)
  return dcnn_a

#INITIALIZE THE MODEL
dcnn_a = Builder_A(Config.MODEL_INPUT)

#PLOT THE MODEL STRUCTURE
print("PLEASE CHECK THE ENTIRE MODEL UP TO THE END")
dcnn_a.summary()
print("Successfully Built!")

# MobileNetV2

build_B = Config.DCNN_B + "_builder"

def Builder_B(model_input):
  builder_b = trainable_model_b(weights='imagenet',
                                include_top=False,
                                input_tensor=model_input)

  for layer in builder_b.layers:
    layer.trainable = False

  for layer in builder_b.layers:
    layer._name = layer.name + '_' + Config.DCNN_B

  for batchnorm in builder_b.layers:
    batchnorm.trainable  = False

# Getting one core block output
  x = builder_b.layers[-39].output

  x = Conv2D(192, 8, padding='valid', activation='selu',  kernel_initializer='lecun_normal')(x)
  x = AveragePooling2D(1,1)(x)
  x = AlphaDropout(0.2)(x)

  dcnn_b = Model(inputs=builder_b.input, outputs=x, name=Config.DCNN_B)
  return dcnn_b

#INITIALIZE THE MODEL
dcnn_b = Builder_B(Config.MODEL_INPUT)

#PLOT THE MODEL STRUCTURE
print("PLEASE CHECK THE ENTIRE MODEL UP TO THE END")
dcnn_b.summary()
print("Successfully Built!")

#ResNet50V2
builder_c = Config.DCNN_C + '_builder'

#TRANSFER LEARNING
def Builder_C(model_input):
    builder_c = trainable_model_c(weights='imagenet',
                                    include_top=False,
                                    input_tensor = model_input)

#PARTIAL LAYER FREEZING
    for layer in builder_c.layers:
        layer.trainable = False

    for layer in builder_c.layers:
        layer._name = layer.name + '_' + Config.DCNN_C

    for BatchNormalization in builder_c.layers:
        BatchNormalization.trainable = False

#LAYER COMPRESSION
    x = builder_c.layers[-117].output #Equivalent to two (2) CORE block deduction.

 #AUXILIARY FUSING LAYER (AuxFL)
    x = Conv2D(192, 6, padding='valid', activation='selu', kernel_initializer='lecun_normal')(x)
    x = AveragePooling2D(3, 3)(x)
    x = AlphaDropout(0.2)(x)

    dcnn_c = Model(inputs=builder_c.input, outputs=x, name=Config.DCNN_C)
    return dcnn_c

#INITIALIZE THE MODEL
dcnn_c = Builder_C(Config.MODEL_INPUT)

#PLOT THE MODEL STRUCTURE
print("PLEASE CHECK THE ENTIRE MODEL UP TO THE END")
dcnn_c.summary()
print("successfully built!")

# Prepare Fusion

dcnn_a = Builder_A(Config.MODEL_INPUT)

dcnn_b = Builder_B(Config.MODEL_INPUT)

dcnn_c = Builder_C(Config.MODEL_INPUT)

models = [dcnn_a,
          dcnn_b,
          dcnn_c]

print("Fusion success!")
print("Ready to connect with its ending layers!")

def mfurecnn_builder(models, model_input):
    outputs = [m.output for m in models]

#INITIAL FUSION LAYER
    y = Add(name='InitialFusionLayer')(outputs)

#FuRB LAYER
    y_bn1 = BatchNormalization()(y)
    y_selu1 = tf.keras.activations.selu(y_bn1)
    y_conv1 = Conv2D(192, 1, kernel_initializer='lecun_normal')(y_selu1)
    y_bn2 = BatchNormalization()(y_conv1)
    y_selu2 = tf.keras.activations.selu(y_bn2)
    y_conv2 = Conv2D(192, 1, kernel_initializer='lecun_normal')(y_selu2)

    y_merge = Add(name='FuRB')([y, y_conv2])

#FINE-TUNING LAYER
    y = GlobalAveragePooling2D()(y_merge)
    y = AlphaDropout(0.5)(y)
    prediction = Dense(5,activation='softmax')(y)
    model = Model(model_input, prediction)
    return model


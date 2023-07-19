# Mutability is Practised


# Mutability is Practised
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

BATCH_SIZE = 8
EPOCHS = 10
OPTIMIZER = Adam(learning_rate=0.00001)
ARCHITECTURE = "MFuReCNN_ALPHA_DO"
DCNN_A = "DCNN_A"   # EfficientNetB0
DCNN_B = "DCNN_B"   # MobileNet
DCNN_C = "DCNN_C"   # ResNet50V2
MAIN_DIR = ""
TRAIN_DATA_DIR = "/content/Train_Test_Folder/train"
VAL_DATA_DIR = "/content/Train_Test_Folder/val"
TEST_DATA_DIR = "/content/Train_Test_Folder/test"
IMG_ROWS = 224
IMG_COLS = 224
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)
MODEL_INPUT = Input(shape = INPUT_SHAPE)
CLASS_NAMES = ["0","1","2","3","4",]
NUM_CLASSES = 5
MODEL_DIR = 'models/'
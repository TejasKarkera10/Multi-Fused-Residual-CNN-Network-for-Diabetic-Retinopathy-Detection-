import Config
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import exposure

class Utils:
    def SAVE_MODEL(FILE, HISTORY):
        with open(FILE + '/' + Config.ARCHITECTURE + '/' + Config.ARCHITECTURE + '.history', 'wb') as MyFile:
            pickle.dump(HISTORY, MyFile)
            print("<====== HISTORY SAVED =======>")

    # Adaptive Histogram Equalization
    def AHE(img):
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
        return img_adapteq

    def PREP_DATA(self):
        train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            rotation_range=45,
                                            shear_range=0.1,
                                            zoom_range=0.1,
                                            height_shift_range=0.1,
                                            width_shift_range=0.1,
                                            fill_mode='constant',
                                            brightness_range=[0.1, 1.0])

        val_data_gen = ImageDataGenerator(rescale=1. / 255)

        test_data_gen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_data_gen.flow_from_directory(
            Config.TRAIN_DATA_DIR,
            target_size=(Config.IMG_ROWS, Config.IMG_COLS),
            batch_size = Config.BATCH_SIZE,
            class_mode = 'categorical',
            seed = 42,
            classes = Config.CLASS_NAMES
        )

        validation_generator = val_data_gen.flow_from_directory(
            Config.VAL_DATA_DIR,
            target_size=(Config.IMG_ROWS, Config.IMG_COLS),
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical',
            seed=42,
            shuffle=False,
            classes=Config.CLASS_NAMES)

        test_generator = test_data_gen.flow_from_directory(
            Config.TEST_DATA_DIR,
            target_size=(Config.IMG_ROWS, Config.IMG_COLS),
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical',
            seed=42,
            shuffle=False,
            classes=Config.CLASS_NAMES)

        # print num samples & Unit Test
        nb_train_samples = len(train_generator.filenames)
        nb_validation_samples = len(validation_generator.filenames)
        nb_test_samples = len(test_generator.filenames)

        if nb_train_samples and nb_validation_samples and nb_test_samples > 0:
            print("Train samples:", nb_train_samples)
            print("Validation samples:", nb_validation_samples)
            print("Test samples:", nb_test_samples)
            print(".... Generators are set! ....")
            print(".... Check if dataset is complete and has no problems before proceeding ....")
            return train_generator, validation_generator, test_generator, nb_train_samples, nb_validation_samples
        else:
            print(".... Issue Detected ....")
            return None, None, None, None, None

        # true labels
        # Y_test = validation_generator.classes
        # test_labels = test_generator.classes

        # Set number of classes automatically
        # num_classes= len(train_generator.class_indices)
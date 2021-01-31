# dcb
import os
import shutil

import numpy as np
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.utils import class_weight


def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, validating_data_dir, test_data_dir,
                                           validate_data_pct=0.3, test_data_pct=0.05):
    """
    
    :param all_data_dir: directory with all files
    :param training_data_dir: directory where we place files we train from
    :param validating_data_dir: directory where we place files on which we valiadate our model
    :param test_data_dir:  directory where we place files where on which we test our build model on
    :param validate_data_pct: % of all files that will be placed in validating_data_dir
    :param test_data_pct: % of remaining files that will be placed in test_data_dir
    :return: 
    """
    # Recreate testing and training directories
    shutil.rmtree(validating_data_dir, ignore_errors=False)
    os.makedirs(validating_data_dir)
    print("Successfully cleaned directory " + validating_data_dir)

    shutil.rmtree(training_data_dir, ignore_errors=False)
    os.makedirs(training_data_dir)
    print("Successfully cleaned directory " + training_data_dir)

    shutil.rmtree(test_data_dir, ignore_errors=False)
    os.makedirs(test_data_dir)
    print("Successfully cleaned directory " + test_data_dir)

    num_training_files = 0
    num_validate_files = 0
    num_test_files = 0

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        validating_data_category_dir = validating_data_dir + '/' + category_name

        testing_data_category_dir = test_data_dir + '/' + category_name
        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(validating_data_category_dir):
            os.mkdir(validating_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < validate_data_pct:
                shutil.copy(input_file, validating_data_dir + '/' + category_name + '/' + file)
                num_validate_files += 1
            else:
                if test_data_dir is not None and np.random.rand(1) < test_data_pct:
                    shutil.copy(input_file, test_data_dir + '/' + category_name + '/' + file)
                    num_test_files += 1
                else:
                    shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                    num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_validate_files) + " validating files.")
    print("Processed " + str(num_test_files) + " testing files.")


def get_weights(train_generator):
    """
    Method to set weights for inbalanced classes. Was not used.
    :param train_generator: generator 
    :return: weights
    """
    classes = list(train_generator.class_indices.values())
    cw = class_weight.compute_class_weight('balanced',
                                           np.unique(classes),
                                           train_generator.classes)
    m = min(cw)
    cw = [(el / m) for el in cw]

    return dict(zip(classes, cw))


# input image size
img_width, img_height = 256, 256

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# building a model
model = Sequential()
model.add(
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer='glorot_normal'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

# we augment the traning data by introducing shear_range, zoom_range and horizontal_flip
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# we rescale all images
valid_datagen = ImageDataGenerator(rescale=1. / 255)
inner_datagen = ImageDataGenerator(rescale=1. / 255)
all_datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 32
# specifying directories with data
train_data_dir = 'train_dir'
validation_data_dir = 'validate_dir'
internal_test_dir = 'testing_dir'

# splitting data in traning, validating and testing set
split_dataset_into_test_and_train_sets('all', train_data_dir, validation_data_dir, internal_test_dir)

# defining generators
train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(img_width, img_height),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)
valid_generator = valid_datagen.flow_from_directory(
    directory=validation_data_dir,
    target_size=(img_width, img_height),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

# we train our model
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=19,
                              verbose=2
                              )
# defining generator for testing images
inner_test_gen = inner_datagen.flow_from_directory(
    directory=internal_test_dir,
    target_size=(img_width, img_height),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)
inner_test_gen.reset()
r = model.evaluate_generator(generator=inner_test_gen)
# we display loss and classification accuracy
print("Loss: " + str(r[0]))
print("Class. acu.: " + str(r[1]))

# Predicting classes for files in dir named test (dir should have a subdirectory where all images are located )
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_dir = 'test'
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(img_width, img_height),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False
)
test_generator.reset()
pred = model.predict_generator(test_generator, verbose=2)
predicted_class_indices = np.argmax(pred, axis=1)

# we label each class as a number from 0, ..., n-1 where n is the number of classes
labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# saving results to a txt file
np.savetxt('res1.txt', predicted_class_indices, fmt='%d')

# for displaying graph of categorical_accuracy, val_categorical_accuracy, val_loss and validation_loss over epochs
if False:
    plt.plot(history.history['categorical_accuracy'], label="training_acc")
    if history.history['val_categorical_accuracy'] is not None:
        plt.plot(history.history['val_categorical_accuracy'], label="validation_acc")

    plt.plot(history.history['loss'], label="training_loss")
    if history.history['val_loss'] is not None:
        plt.plot(history.history['val_loss'], label="validation_loss")
    plt.legend(loc='upper left')
    plt.show()

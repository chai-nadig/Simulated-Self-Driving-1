import csv
import cv2

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# The dataset to train
folder = './train4'


# This helper function reads all the lines in `driving_log.csv`
# and returns them as an array.
def read_dataset():
    lines = []

    with open('{}/driving_log.csv'.format(folder)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines


# A generator is used to read images in batches.
# The default `batch_size` is 32.
# For every line in `driving_log.csv`, this generator
# produces 6 images.
#
# The first three images are from the center, left and right cameras.
# These three images are then flipped to create augmented images
# giving totally six images.
#
# The measurements for the left and right images are also corrected
# using `correction`.
#
# The final batch is shuffled and returned
def generator(samples, batch_size=32, correction=0.3):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            batch_images = []
            batch_measurements = []
            for line in batch_samples:
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = '{}/IMG/{}'.format(folder, filename)
                    image = cv2.imread(current_path)
                    batch_images.append(image)
                    measurement = float(line[3]) + (0 if i == 0 else correction if i == 1 else -correction)
                    batch_measurements.append(measurement)

            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(batch_images, batch_measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield shuffle(X_train, y_train)


# Read all lines and do a 20% split for training and validation
all_samples = read_dataset()
train_samples, validation_samples = train_test_split(all_samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Here we build the neural network
model = Sequential()

# Normalize the pixels and mean center them
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

# Crop the frames to ignore irrelevant pixel
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# The following lines build a neural network as proposed by Nvidia
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')
exit()

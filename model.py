import csv
import cv2
import numpy as np
from random import shuffle
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt


driving_log_path = 'my_data/driving_log.csv'
image_path = 'my_data/IMG/'
lines = []
with open(driving_log_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


def generator(samples, batch_size=32):
    steering_correction = 0.25
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for camera_index in range(3):
                    source_path = batch_sample[camera_index]
                    filename = source_path.split('/')[-1]
                    current_path = image_path + filename
                    image = cv2.imread(current_path)
                    if image is None:
                        continue
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                try:
                    measurement = float(batch_sample[3])
                except ValueError:
                    print("Measurement Error: ", batch_sample[3])
                measurements.append(measurement)
                measurements.append(measurement + steering_correction)
                measurements.append(measurement - steering_correction)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
# model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.4))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.4))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Dropout(0.4))
model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation="relu"))
model.add(Dropout(0.4))
model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation="relu"))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
#                    nb_val_samples=len(validation_samples), nb_epoch=3)
batch_size = 32
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples)/batch_size, epochs=3, verbose=1)

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')

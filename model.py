import keras
from keras.layers import Input, Flatten, Dense, Activation, Lambda, Conv2D, MaxPooling2D, Dropout
import cv2
import csv
import numpy as np
import sklearn
import math

TRAIN_DATA_PATH = '/home/carnd/data'
VALID_DATA_PATH = '/home/carnd/valid_data'
IMAGE_SHAPE =(160, 320, 3)
BATCH_SIZE = 64

class data_gen():

    def __init__(self, data_path, batch_size=128):
        self.data_path = data_path
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.reader = csv.reader(open('/'.join([self.data_path, 'driving_log.csv'])))
        self.reader.__next__()

    def __iter__(self):
        return self

    def __next__(self):

        images = []
        angles = []
       
        try:
            while len(images) < self.batch_size:
                line = self.reader.__next__()
                f = line[0].split('/')[-1]
                img = cv2.imread('/'.join([self.data_path, 'IMG', f]))
                angle = float(line[3])

                images.append(img)
                angles.append(angle)

                # Augment by flipping images.
                images.append(np.fliplr(img))
                angles.append(-angle)

        except StopIteration:
           self.reset()

        data, labels = np.array(images, dtype=np.float32), np.array(angles, dtype=np.float32)
        return sklearn.utils.shuffle(data, labels)

# Have to account for augmentation here.
def count_dataset(data_path):
    csvfile = open('/'.join([data_path, 'driving_log.csv']))
    next(csvfile)
    num = 0
    for line in csv.reader(csvfile):
        num += 1
    return num * 2

def network():
    net = keras.models.Sequential()

    # Crop off top third of image. Result is 110x320 image.
    net.add(Lambda(lambda x: x[:,50:,...], input_shape=IMAGE_SHAPE))

    # Normalize.
    net.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # Output: 100x300x16  
    conv1 = Conv2D(nb_filter=16, nb_row=11, nb_col=21, subsample=(1,1), border_mode='valid', bias=True)
    net.add(conv1) 
    net.add(Activation('relu'))

    # Output: 50x150x16
    net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))

    # Output 42x142x32
    conv2 = Conv2D(nb_filter=32, nb_row=9, nb_col=9, subsample=(1,1), border_mode='valid', bias=True)
    net.add(conv2)
    net.add(Activation('relu'))

    # Outout 21x71x32
    net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))

    # 18x68x64
    conv3 = Conv2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(1,1), border_mode='valid', bias=True)
    net.add(conv3)
    net.add(Activation('relu'))

    net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))

    net.add(Flatten())

    net.add(Dense(256))
    net.add(Activation('relu'))

    net.add(Dense(256))
    net.add(Activation('relu'))

    net.add(Dense(256))
    net.add(Activation('relu'))

    net.add(Dense(1))
    return net

def main():
    net = network()

    dataset_size = count_dataset(TRAIN_DATA_PATH)
    validation_set_size = count_dataset(VALID_DATA_PATH)
    train_data = data_gen(TRAIN_DATA_PATH, BATCH_SIZE)
    valid_data = data_gen(VALID_DATA_PATH, BATCH_SIZE)

    print("Size of training data: {}.".format(dataset_size))
    print("Size of validation data: {}.".format(validation_set_size))

    net.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001))
    net.fit_generator(train_data, samples_per_epoch=dataset_size, nb_epoch=10, verbose=2, validation_data=valid_data, nb_val_samples=validation_set_size)

    net.summary()
    net.save('net.h5')

    # Print some predictions.
    x, y = data_gen(TRAIN_DATA_PATH, BATCH_SIZE).__next__()

    pred = net.predict(x)

    print("Some predictions:")
    print(y[:10])
    print(pred[:10])



if __name__ == '__main__':
    main()


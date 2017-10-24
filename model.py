import argparse
import keras
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Activation, Lambda, Conv2D, MaxPooling2D, Dropout
import cv2
import csv
import numpy as np
import sklearn
import math

TRAIN_DATA_PATH = '/home/carnd/clean_data'
VALID_DATA_PATH = '/home/carnd/valid_data'
IMAGE_SHAPE =(160, 320, 3)
BATCH_SIZE = 128
OFF_CENTER_CORRECTION = 0.2 # Steering angle offset for left & right camera frames.

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
        speeds = []

        try:
            while len(images) < self.batch_size:
                line = self.reader.__next__()
                f_c = line[0].split('/')[-1]
                f_l = line[1].split('/')[-1]
                f_r = line[2].split('/')[-1]

                img_c = cv2.imread('/'.join([self.data_path, 'IMG', f_c]))
                img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)

                img_l = cv2.imread('/'.join([self.data_path, 'IMG', f_l]))
                img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)

                img_r = cv2.imread('/'.join([self.data_path, 'IMG', f_r]))
                img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

                angle = float(line[3])
                speed = float(line[6])

                # Augment by adding left & right camera images with offset angle.
                images.append(img_c)
                angles.append(angle)
                speeds.append(speed)

                images.append(img_l)
                angles.append(angle + OFF_CENTER_CORRECTION)
                speeds.append(speed)

                images.append(img_r)
                angles.append(angle - OFF_CENTER_CORRECTION)
                speeds.append(speed)

                # Augment by flipping images.
                images.append(np.fliplr(img_c))
                angles.append(-angle)
                speeds.append(speed)

                images.append(np.fliplr(img_l))
                angles.append(-angle - OFF_CENTER_CORRECTION)
                speeds.append(speed)

                images.append(np.fliplr(img_r))
                angles.append(-angle + OFF_CENTER_CORRECTION)
                speeds.append(speed)

        except StopIteration:
           self.reset()

        images, speeds, angles = sklearn.utils.shuffle(np.array(images, dtype=np.float32), np.array(speeds), np.array(angles))
        return [images, speeds], angles

# Have to account for augmentation here.
def count_dataset(data_path):
    csvfile = open('/'.join([data_path, 'driving_log.csv']))
    next(csvfile)
    num = 0
    for line in csv.reader(csvfile):
        num += 1
    return num * 6

def network():

    image = Input(IMAGE_SHAPE, dtype='float32', name='image_input')
    speed = Input((1,), dtype='float32', name='speed_input')

    # Crop off top third of image. Result is 110x320 image.
    pre1 = Lambda(lambda x: x[:,50:,...], name='crop')(image)

    # Normalize.
    pre2 = Lambda(lambda x: (x / 255.0) - 0.5, name='normalize')(pre1)

    # Output: 100x300x16  
    conv1 = Conv2D(nb_filter=16, nb_row=11, nb_col=21, subsample=(1,1), border_mode='valid', bias=True)(pre2)
    conv1 = Activation('relu')(conv1)

    # Output: 50x150x16
    pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid')(conv1)

    # Output 42x142x32
    conv2 = Conv2D(nb_filter=32, nb_row=9, nb_col=9, subsample=(1,1), border_mode='valid', bias=True)(pool1)
    conv2 = Activation('relu')(conv2)

    # Outout 21x71x32
    pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid')(conv2)

    conv3 = Conv2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(1,1), border_mode='valid', bias=True)(pool2)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid')(conv3)

    conv4 = Conv2D(nb_filter=128, nb_row=4, nb_col=5, subsample=(1,1), border_mode='valid', bias=True)(pool3)
    conv4 = Activation('relu')(conv4)

    pool4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid')(conv4)

    flat = Flatten()(pool4)
    cat = keras.layers.merge([flat, speed], mode='concat')

    fc1 = Dense(128)(cat)
    fc1 = Activation('relu')(fc1)

    fc2 = Dense(64)(fc1)
    fc2 = Activation('relu')(fc2)

    fc3 = Dense(32)(fc2)
    fc3 = Activation('relu')(fc3)

    out = Dense(1)(fc3)
    return Model(input=[image, speed], output=[out])

def main(pretrained):

    if pretrained == '':
        net = network()
    else:
        net = load_model(pretrained)

    dataset_size = count_dataset(TRAIN_DATA_PATH)
    validation_set_size = count_dataset(VALID_DATA_PATH)
    train_data = data_gen(TRAIN_DATA_PATH, BATCH_SIZE)
    valid_data = data_gen(VALID_DATA_PATH, BATCH_SIZE)

    print("Size of training data: {}.".format(dataset_size))
    print("Size of validation data: {}.".format(validation_set_size))

    net.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0001))
    net.fit_generator(train_data, samples_per_epoch=dataset_size, nb_epoch=5, verbose=2, validation_data=valid_data, nb_val_samples=validation_set_size)

    net.summary()
    print("Writing model out to 'net.h5'")
    net.save('net.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Training')
    parser.add_argument('-p', '--pretrained', type=str, default = '', help='Path to h5 file for pre-trained model to fine tune.')
    args = parser.parse_args()

    main(args.pretrained)

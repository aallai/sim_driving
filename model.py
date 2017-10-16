import keras
from keras.layers import Input, Flatten, Dense, Activation, Lambda, Conv2D, MaxPooling2D, Dropout
import cv2
import csv
import numpy as np
from sklearn.utils import shuffle

TRAIN_DATA_PATH = '/home/carnd/data'
VALID_DATA_PATH = '/home/carnd/valid_data'
IMAGE_SHAPE =(160, 320, 3)
BATCH_SIZE = 64

# TODO: use other features.
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
        return shuffle(data, labels)

def count_dataset(data_path):
    csvfile = open('/'.join([data_path, 'driving_log.csv']))
    next(csvfile)
    num = 0
    for line in csv.reader(csvfile):
        num += 1
    return num

# TODO: improve architecture.
def network():
    net = keras.models.Sequential()

    # Crop off top third of image. Result is 110x320 image.
    net.add(Lambda(lambda x: x[:,50:,...], input_shape=IMAGE_SHAPE))

    # Normalize.
    net.add(Lambda(lambda x: (x / 255.0) - 0.5))
    
    # Merge color channels
    #net.add(Conv2D(nb_filter=1, nb_row=1, nb_col=1, subsample=(1,1), border_mode='valid'))

    # Output: 100x300x16  
    conv1 = Conv2D(nb_filter=16, nb_row=11, nb_col=21, subsample=(1,1), border_mode='valid', bias=True)
    net.add(conv1) 
    net.add(Activation('relu'))

    # Output: 25x75x16
    net.add(MaxPooling2D(pool_size=(4,4), strides=(4,4), border_mode='valid'))

    # Output 20x70x32
    conv2 = Conv2D(nb_filter=32, nb_row=6, nb_col=6, subsample=(1,1), border_mode='valid', bias=True)
    net.add(conv2)
    net.add(Activation('relu'))

    net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))

    net.add(Flatten())
    net.add(Dropout(p=0.7))

    net.add(Dense(256))
    net.add(Activation('relu'))
    net.add(Dropout(p=0.7))

    net.add(Dense(256))
    net.add(Activation('relu'))
    net.add(Dropout(p=0.7))

    net.add(Dense(1))
    return net

def main():
    net = network()

    dataset_size = count_dataset(TRAIN_DATA_PATH)
    validation_set_size = count_dataset(VALID_DATA_PATH)
    train_data = data_gen(TRAIN_DATA_PATH, BATCH_SIZE)
    valid_data = data_gen(VALID_DATA_PATH, BATCH_SIZE)

    print("Size of training data (without 2x augmentation): {}.".format(dataset_size))
    print("Size of validation data: {}.".format(validation_set_size))

    net.compile(loss='mse', optimizer='adam')
    net.fit_generator(train_data, samples_per_epoch=dataset_size, nb_epoch=50, verbose=2, validation_data=valid_data, nb_val_samples=validation_set_size)

    for layer in net.layers:
        print(layer, layer.trainable)

    net.summary()
    net.save('net.h5')

if __name__ == '__main__':
    main()


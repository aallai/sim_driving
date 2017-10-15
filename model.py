import keras
from keras.layers import Flatten, Dense, Activation 
import cv2
import csv
import numpy as np

DATA_PATH = '/home/carnd/data'

# TODO: use other features.
def load_data():
    images = []
    angles = []

    with open('/'.join([DATA_PATH, 'driving_log.csv'])) as csvfile:
        next(csvfile) # Skip header.
        reader = csv.reader(csvfile)
        for line in reader:
            images.append(cv2.imread('/'.join([DATA_PATH,line[0]])))
            angles.append(float(line[3]))

    return np.array(images, dtype=np.float32), np.array(angles)

# TODO: improve architecture.
def network(x):
    net = keras.models.Sequential()
    net.add(Flatten(input_shape=x.shape[1:]))
    net.add(Dense(256))
    net.add(Activation('relu'))
    net.add(Dense(1))
    return net

def main():
    images, angles = load_data()

    print(images.shape)
    print(images[0])

    net = network(images)

    net.compile(loss='mse', optimizer='adam')
    net.fit(images, angles, nb_epoch=100, batch_size=128, validation_split=0.2, shuffle=True, verbose=2)
    net.save('net.h5')

if __name__ == '__main__':
    main()

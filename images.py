import cv2
import csv
import argparse
import numpy as np

# Reproduce the augmentation from model.py, for writeup purposes.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Generation')
    parser.add_argument('-d', '--data', type=str, default = '', help='Path to recorded driving data to use for image generation.')
    args = parser.parse_args()

    reader = csv.reader(open('/'.join([args.data, 'driving_log.csv'])))
    
    # Loop through a few lines to find an interesting image.
    for i in range(150):
        reader.__next__()

    line = reader.__next__()
    f = line[0].split('/')[-1]

    img = cv2.imread('/'.join([args.data, 'IMG', f]))
    img_flipped = np.fliplr(img)

    cv2.imwrite('data_sample.png', img)
    cv2.imwrite('flipped_data_sample.png', img_flipped)
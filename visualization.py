from keras.utils.visualize_util import plot
from keras.models import load_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Diagram')
    parser.add_argument('-m', '--model', type=str, help='Path to h5 model file.')
    args = parser.parse_args()

    net = load_model(args.model)

    print("Writing out 'model.png'.")
    plot(net, to_file='model.png')
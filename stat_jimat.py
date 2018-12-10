import os
import sys
import threading
import glob
import random
import argparse
import numpy as np

from python_utils.preprocessing import data_loader

def main(args=None):

    # parse arguments
    #if args is None:
    #    args = sys.argv[1:]

    path_to_raw = './jimat/data'
    train_len = int(len(os.listdir(os.path.join(path_to_raw, 'train'))) / 2)
    val_len = int(len(os.listdir(os.path.join(path_to_raw, 'val'))) / 2)
    print('training: '+str(train_len))
    print('val: '+str(val_len))

    train_data, train_label = data_loader(
        datadir=os.path.join(path_to_raw, 'train'), input_size=(352,352), nb_classes=2, separator='_', padding=True)
    val_data, val_data = data_loader(
        datadir=os.path.join(path_to_raw, 'val'), input_size=(352,352), nb_classes=2, separator='_', padding=True)

    images = np.concatenate((train_data, val_data), axis=0)
    # Normalize pixel values in images
    images = images / 255.
    img_mean = images.mean()
    img_std = images.std()

    np.savetxt('DATA_MEAN', img_mean, delimiter=',')
    np.savetxt('DATA_STD', img_std, delimiter=',')

if __name__ == '__main__':
    main()





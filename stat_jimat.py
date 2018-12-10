import os
import sys
import threading
import glob
import random
import argparse

from python_utils.preprocessing import data_loader

def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    train_len = int(len(os.listdir(os.path.join(args.path_to_raw, 'train'))) / 2)
    val_len = int(len(os.listdir(os.path.join(args.path_to_raw, 'val'))) / 2)
    train_data, train_label = data_loader(
        datadir=os.path.join(args.path_to_raw, 'train'), input_size=(352,352), nb_classes=2, separator='_', padding=True)
    val_data, val_data = data_loader(
        datadir=os.path.join(args.path_to_raw, 'val'), input_size=(352,352), nb_classes=2, separator='_', padding=True)

    images = np.stack([train_data, val_data])
    # Normalize pixel values in images
    images = images / 255.
    img_mean = images.mean()
    img_std = images.std()

    np.savetxt('DATA_MEAN', img_mean, delimiter=',')
    np.savetxt('DATA_STD', img_std, delimiter=',')

if __name__ == '__main__':
    main()





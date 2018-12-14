import os
import sys
import threading
import glob
import random
import argparse

import keras
from keras.models import Model
from keras.layers import *
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.utils import multi_gpu_model
from python_utils.preprocessing import data_generator_s31
from python_utils.callbacks import callbacks

from tiramisu.model import create_tiramisu
from alt_model_checkpoint import AltModelCheckpoint

def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for training The One Hundred Layers Tiramisu network.')

    parser.add_argument('--output_path',
                        help='Path for saving a training model as a *.h5 file. Default is models/new_tiramisu.h5',
                        default='models/jimat_tiramisu.h5')
    parser.add_argument('--path_to_raw',
                        help='Path to raw images used for training. Default is jimat/data/',
                        default='jimat/data')
    parser.add_argument('--image_size',
                        help='Size of the input image. Default is [224, 224]',
                        default=(352, 352))
    parser.add_argument('--log_dir',
                        help='Path for storing tensorboard logging. Default is logging/',
                        default='logging/')
    parser.add_argument('--no_epochs',
                        type=int,
                        help='Defines number of epochs used for training. '
                             'Default: 500',
                        default=500)
    parser.add_argument('--nb_classes',
                        type=int,
                        help='Defines number of classes for training. '
                             'Default: 2',
                        default=2)
    parser.add_argument('--batch_size',
                        type=int,
                        help='Defines batch size for training. '
                             'Default: 16',
                        default=16)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Defines learning rate used for training. '
                             'Default: 1e-3',
                        default=1e-3)
    parser.add_argument('--patience',
                        type=int,
                        help='Defines patience for early stopping. '
                             'Default: 150',
                        default=150)
    parser.add_argument('--path_to_model_weights',
                        help='Path to saved model weights if training should be resumed. '
                             'Default: models/new_tiramisu.h5',
                        default='models/new_tiramisu.h5')
    parser.add_argument('--train_from_zero',
                        type=bool,
                        help='Boolean, defines if training from scratch or resuming from saved h5 file. '
                             'Default: True',
                        default=True)
    parser.add_argument('--use_multi_gpu',
                        type=bool,
                        help='Defines if use multiple gpu. '
                             'Default: True',
                        default=True)

    return parser.parse_args(args)


def main(args=None):

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' # str(args.gpu)

    img_size = args.image_size

    train_len = int(len(os.listdir(os.path.join(args.path_to_raw, 'train'))) / 2)
    val_len = int(len(os.listdir(os.path.join(args.path_to_raw, 'val'))) / 2)

    train_generator = data_generator_s31(
        datadir=os.path.join(args.path_to_raw, 'train'), batch_size=args.batch_size, input_size=img_size, nb_classes=args.nb_classes, separator='_', padding=True)
    val_generator = data_generator_s31(
        datadir=os.path.join(args.path_to_raw, 'val'), batch_size=args.batch_size, input_size=img_size, nb_classes=args.nb_classes, separator='_', padding=True)

    # config network architecture
    #nb_dense_block=6
    #growth_rate=16
    #nb_filter=48
    #nb_layers_per_block=[4, 5, 7, 10, 12, 15]
    nb_layers_per_block=[5, 5, 5, 5, 5, 5]
    input_shape = img_size + (3,)
    with tf.device('/cpu:0'):
        img_input = Input(shape=input_shape)
        x = create_tiramisu(args.nb_classes, img_input, nb_layers_per_block=nb_layers_per_block)
        model = Model(img_input, x)
        print(model.summary())

    if not args.train_from_zero:
        model.load_weights(args.path_to_model_weights)

    if args.use_multi_gpu:
        #model = multi_gpu_model(model, gpus=4, cpu_merge=False)
        parallel_model = multi_gpu_model(model, gpus=4)

    parallel_model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.RMSprop(args.learning_rate, decay=1-0.99995), metrics=["accuracy"])

    logging = TensorBoard(log_dir=args.log_dir)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=args.patience, verbose=1, mode='auto')
    #checkpoint = ModelCheckpoint(args.log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
    #                             monitor='val_loss', save_weights_only=False, save_best_only=True)
    checkpoint = AltModelCheckpoint(args.log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                    model,
                                    monitor='val_loss', save_weights_only=False, save_best_only=True)

    #model.fit_generator(train_generator, len(train_set), args.no_epochs, verbose=2,
    #                    validation_data=test_generator, validation_steps=len(val_set),
    #                    callbacks=[logging, checkpoint, early_stopping])
    #history = model.fit(train_set, train_labels, batch_size=args.batch_size, epochs=args.no_epochs, verbose=2,
    #                callbacks=[logging, checkpoint, early_stopping], validation_data=(val_set, val_labels),
    #                class_weight=class_weighting, shuffle=True)
    history = parallel_model.fit_generator(generator=train_generator, epochs=500, verbose=2, steps_per_epoch=500,
                                           validation_data=val_generator, validation_steps=31,
                                           callbacks=[logging, checkpoint, early_stopping])

    model.save(args.output_path)

    print(history.history.keys())

if __name__ == '__main__':
    main()




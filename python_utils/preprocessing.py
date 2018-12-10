import os
import random
import numpy as np
from scipy.misc import imresize, imread
from scipy.ndimage import zoom
from collections import defaultdict

#DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])
DATA_MEAN = np.array([[[.68, .779, .939]]])
DATA_STD = np.array([[[.68, .779, .939]]])

def preprocess_img(img):
    #img = imresize(img, input_shape)
    #img = img - DATA_MEAN
    #img = img[:, :, ::-1]
    
    # Normalize pixel values in images
    img = img / 255.
    img -= DATA_MEAN #img.mean()
    img /= DATA_STD #img.std()

    # # #
    img.astype('float32')
    return img

def update_inputs(batch_size = None, input_size = None, num_classes = None):
  return np.zeros([batch_size, input_size[0], input_size[1], 3], dtype=float32), \
    np.zeros([batch_size, input_size[0], input_size[1], num_classes], dtype=float32)

def data_generator_s31(datadir='', nb_classes = None, batch_size = None, input_size=None, separator='_', padding=True):
  if not os.path.exists(datadir):
    print("ERROR!The folder is not exist")
  #os.listdir(os.path.join(datadir, "train"))
  data = defaultdict(dict)
  img_names = os.listdir(datadir)
  img_names.sort()
  nb_pairs = int(len(img_names) / 2)

  for img_name in img_names:
    image_nmb = '_'.join(img_name.split(separator)[0:2])
    image_type = int(img_name.split(separator)[2])
    if image_type==0:
      data[image_nmb]['image'] = img_name
    elif image_type==1:
      data[image_nmb]['anno'] = img_name
    else:
      print("ERROR!The image type is not exist")
      print(str(image_nmb)+'\t'+image_type)

  for k, v in data.iteritems():
    if 'image' not in v or 'anno' not in v:
      print("ERROR!The image is not paired up")
      print(k)

  values = data.values()
  #shuffle and gen values
  random.shuffle(values)
  return generate(values, nb_classes, batch_size, input_size, datadir)

def generate(values, nb_classes, batch_size, input_size, datadir, padding):
  while 1:
    nb_pairs = len(values)
    random.shuffle(values)
    images, labels = update_inputs(batch_size=batch_size,
       input_size=input_size, num_classes=nb_classes)
    for i, d in enumerate(values):
      img = imread(os.path.join(datadir, d['image']), mode='RGB')
      y = imread(os.path.join(datadir, d['anno']), mode='L')
      h, w = input_size
      ###########
      if padding:
        img = np.pad(img, (((h-img.shape[0])/2, (h-img.shape[0])/2), ((w-img.shape[1])/2, (w-img.shape[1])/2), (0,0)), 'constant', constant_values=(0))
        y = np.pad(y, (((h-y.shape[0])/2, (h-y.shape[0])/2), ((w-y.shape[1])/2, (w-y.shape[1])/2)), 'constant', constant_values=(0))
      else:
        img = imresize(img, input_size)
        y = zoom(y, (1.*h/y.shape[0], 1.*w/y.shape[1]), order=1, prefilter=False)
      y = (np.arange(nb_classes) == y[:,:,None]).astype('float32')
      assert y.shape[2] == nb_classes
      images[i % batch_size] = img
      labels[i % batch_size] = y
      #if (i + 1) % batch_size == 0 or (i + 1) == nb_pairs:
      if (i + 1) % batch_size == 0:
        ######
        images = preprocess_img(images)
        yield images, labels
        images, labels = update_inputs(batch_size=batch_size,
          input_size=input_size, num_classes=nb_classes)


def data_loader(datadir='', nb_classes = None, input_size=None, separator='_', padding=True):
  if not os.path.exists(datadir):
    print("ERROR!The folder is not exist")
  #os.listdir(os.path.join(datadir, "train"))
  data = defaultdict(dict)
  img_names = os.listdir(datadir)
  img_names.sort()

  for img_name in img_names:
    image_nmb = '_'.join(img_name.split(separator)[0:2])
    image_type = int(img_name.split(separator)[2])
    if image_type==0:
      data[image_nmb]['image'] = img_name
    elif image_type==1:
      data[image_nmb]['anno'] = img_name
    else:
      print("ERROR!The image type is not exist")
      print(str(image_nmb)+'\t'+image_type)

  for k, v in data.iteritems():
    if 'image' not in v or 'anno' not in v:
      print("ERROR!The image is not paired up")
      print(k)

  values = data.values()
  
  #shuffle and gen values
  random.shuffle(values)
  values = values[1:10000]

  nb_pairs = len(values)
  images = np.zeros([nb_pairs, input_size[0], input_size[1], 3], dtype=float32)
  labels = np.zeros([nb_pairs, input_size[0], input_size[1], nb_classes], dtype=float32)
  for i, d in enumerate(values):
    img = imread(os.path.join(datadir, d['image']), mode='RGB')
    y = imread(os.path.join(datadir, d['anno']), mode='L')
    h, w = input_size
    ###########
    if padding:
      img = np.pad(img, (((h-img.shape[0])/2, (h-img.shape[0])/2), ((w-img.shape[1])/2, (w-img.shape[1])/2), (0,0)), 'constant', constant_values=(0))
      y = np.pad(y, (((h-y.shape[0])/2, (h-y.shape[0])/2), ((w-y.shape[1])/2, (w-y.shape[1])/2)), 'constant', constant_values=(0))
    else:
      img = imresize(img, input_size)
      y = zoom(y, (1.*h/y.shape[0], 1.*w/y.shape[1]), order=1, prefilter=False)
    y = (np.arange(nb_classes) == y[:,:,None]).astype('float32')
    assert y.shape[2] == nb_classes
    images[i] = img
    labels[i] = y
  
  ######
  #images = preprocess_img(images)
  return images, labels


from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LSTM, RepeatVector
from keras.layers.wrappers import *
from keras.layers.core import *
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import tensorflow as tf

import imageio
import cv2
import numpy as np


max_title_length = 20
number_of_chars = 200
latent_dim = 800
max_title_len = 20

input_sentence = Input(shape=(max_title_length, number_of_chars), dtype='int32')

tofloat = Lambda(function=lambda x: tf.to_float(x))(input_sentence)
encoder = LSTM(latent_dim, activation='tanh')(tofloat)

decoder = RepeatVector(max_title_len)(encoder)
decoder = LSTM(number_of_chars, return_sequences=True, activation='tanh')(decoder)
autoencoder = Model(input=input_sentence, output=decoder)

autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
plot_model(autoencoder, to_file='model.png', show_shapes=True)

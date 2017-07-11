from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LSTM, RepeatVector
from keras.layers.wrappers import *
from keras.layers.core import *
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt

import imageio
import cv2
import numpy as np

LOG_DIR = "../../tensorboard/log/"
EPOCH = 500


def generate_movies(n_samples=1200):
    np.random.seed(19921010)

    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, row, col, 1), dtype=np.float)

    for i in range(n_samples):
        # Add 1 to 4 moving squares
        # n = np.random.randint(1, 5)

        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion

            # Size of the square
            w = np.random.randint(2, 4)
            shifted_movies[i, xstart - w: xstart + w,
                               ystart - w: ystart + w, 0] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::,  20:60, 20:60, ::]
    shifted_movies = shifted_movies[::,  20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return  shifted_movies, noisy_movies


input_img = Input(shape=(40, 40, 1))  # adapt this if using `channels_first` image data format

# flat = Flatten()(input_img)
# d1 = Dense(800, activation='relu')(flat)
# d2 = Dense(800, activation='relu')(d1)
# d3 = Dense(1600, activation='sigmoid')(d2)
# decoded = Reshape((40,40,1))(d3)

conv1 = Conv2D(6, (3, 3), activation='relu', padding='same')(input_img)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1)
conv2 = Conv2D(3, (3, 3), activation='relu', padding='same')(pool1)
encoded = MaxPooling2D((2, 2), padding='same')(conv2)

# d2 = Dense(800, activation='relu')(d1)


conv3 = Conv2D(3, (3, 3), activation='relu', padding='same')(encoded)
up1 = UpSampling2D((2, 2))(conv3)
conv4 = Conv2D(6, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv4)

decoded = Dense(1, activation='sigmoid')(up2)
# decoded = Conv2D(1, 3, 3, activation='sigmoid', border_mode='same')(up2)



# at this point the representation is (4, 4, 8) i.e. 128-dimensional
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
# encoder = Model(input_img, encoded)

autoencoder.compile(optimizer='RMSprop', loss='mean_squared_error')
plot_model(autoencoder, to_file='cnn-model.png', show_shapes=True)


x_train, noisy_movies = generate_movies(n_samples=10000)

autoencoder.fit(x_train[:9500], x_train[:9500],
                epochs=200,
                batch_size=100,
                shuffle=True,
                validation_split=0.05,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions

for i in range(10):

    which = 9800+i
    track = x_train[which][ ::, ::, ::]
    track2 = autoencoder.predict(track[np.newaxis, ::, ::, ::])
    track2 = track2[0][::, ::, ::]

    # track2 = autoencoder.predict(track)

    # for j in range(16):
    #     new = new_pos[::, -1, ::, ::, ::]
    #     track = np.concatenate((track, new), axis=0)

    # And then compare the predictions
    # to the ground truth
    # track2 = shifted_movies[which][::, ::, ::, ::]

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(131)
    # if i >= 7:
    # ax.text(1, 3, 'Ground Truth', fontsize=20, color='w')
    ax.text(1, 3, 'Ground Truth', fontsize=20, color='white')
    # else:
        # ax.text(1, 3, 'Inital trajectory', fontsize=20)
    toplot = track[::, ::, 0]
    plt.imshow(toplot.reshape(40,40) ,vmin=0, vmax=1, cmap='jet', aspect='auto')
    plt.gray()
    plt.colorbar()

    ax = fig.add_subplot(132)
    plt.text(1, 3, 'Recovered(Same)', fontsize=20, color='white')
    toplot = track2[::, ::, 0]

    plt.imshow(toplot.reshape(40,40),vmin=0, vmax=1, cmap='jet', aspect='auto')
    plt.gray()
    plt.colorbar()

    ax = fig.add_subplot(133)
    plt.text(1, 3, 'Recovered(Scaled)', fontsize=20, color='white')
    toplot = track2[::, ::, 0]

    plt.imshow(toplot.reshape(40,40)  ,cmap='jet', aspect='auto')
    plt.gray()
    plt.colorbar()

    plt.savefig('static_animate_%i.png'% i) 




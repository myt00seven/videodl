# This is a sequential classification

""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
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
from keras import backend as K

import os
import time
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt

import imageio
import cv2
import numpy as np

GENERATE_DATA = 1 
# 1 if generate aritificial data, 0 if use UCF101 data

LOG_DIR = "../../tensorboard/log/"
EPOCH = 100
sequenceLength = 5
setup_name = "clrmvsq_simple_vgg_a"
N_SAMPLES = 1500
BATCHSIZE = 5


# seq = Sequential()

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    input_shape=(None, 40, 40, 1),
#                    padding='same', activation='relu', return_sequences=True))
# seq.add(BatchNormalization())
# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', activation='relu', return_sequences=True))
# seq.add(BatchNormalization())
# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', activation='relu', return_sequences=True))
# seq.add(BatchNormalization())
# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', activation='relu', return_sequences=True))
# seq.add(BatchNormalization())
# seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
#                activation='sigmoid',
#                padding='same', activation='relu', data_format='channels_last'))
# seq.compile(loss='binary_crossentropy', optimizer='adadelta')

###################################
# Data Loading
###################################

# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.
def set_bound(pos):
    if pos <0:
        return 0
    elif pos>224+40:
        return 224+39
    else:
        return pos

def generate_movies(n_samples=N_SAMPLES, n_frames=sequenceLength):
    np.random.seed(19921010)

    row = 224 + 40
    col = 224 + 40
    # noisy_movies = np.zeros((n_samples, n_frames, row, col, 3), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 3),dtype=np.float)

    for i in range(n_samples):
        # Add 1 to 4 moving squares
        # n = np.random.randint(1, 5)

        if i%100==0:
                print("Generating %d th data." % i )

        # Add 10 to 20 moving squares
        # n = np.random.randint(10, 21)
        n = np.random.randint(4, 11)

        for j in range(n):

            # Initial position
            xstart = np.random.randint(20, 20+224)
            ystart = np.random.randint(20, 20+224)
            # Direction of motion
            # directionx = 0
            directionx = np.random.randint(0, 21) - 10
            # directiony = 0
            directiony = np.random.randint(0, 21) - 10

            # Size of the square
            # w = np.random.randint(2, 4)
            w = np.random.randint(3, 10)

            color_r = np.random.randint(100,205) /255.0
            color_g = np.random.randint(100,205) /255.0
            color_b = np.random.randint(100,205) /255.0

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                # noisy_movies[i, t, x_shift - w: x_shift + w,
                             # y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                # if np.random.randint(0, 2):
                #     noise_f = (-1)**np.random.randint(0, 2)
                #     noisy_movies[i, t,
                #                  x_shift - w - 1: x_shift + w + 1,
                #                  y_shift - w - 1: y_shift + w + 1,
                #                  0] += noise_f * 0.1
                
                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)

                shifted_movies[i, t, set_bound(x_shift - w): set_bound(x_shift + w), set_bound(y_shift - w): set_bound(y_shift + w), 0] = color_r
                shifted_movies[i, t, set_bound(x_shift - w): set_bound(x_shift + w), set_bound(y_shift - w): set_bound(y_shift + w), 1] = color_g
                shifted_movies[i, t, set_bound(x_shift - w): set_bound(x_shift + w), set_bound(y_shift - w): set_bound(y_shift + w), 2] = color_b

    # Cut to a 40x40 window
    # noisy_movies = noisy_movies[::, ::, 20:20+224, 20:20+224, ::]
    shifted_movies = shifted_movies[::, ::, 20:20+224, 20:20+224, ::]
    # noisy_movies[noisy_movies >= 1] = 1
    # shifted_movies[shifted_movies >= 1] = 1
    return shifted_movies

def get_ucf_data(n_samples):
    return 


def DenseNetwork(inputs):
    x = Dense(5, activation='relu')(inputs)
    x = Dense(1, activation='sigmoid')(x)
    return x

def MyCNNthenDeCNN(inputs):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same', activation='relu')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same', activation='relu')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
    return decoded

def MyCNN(inputs):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same', activation='relu')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same', activation='relu')(x)
    return x

def MyDeCNN(inputs):
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
    return decoded

def plot_val(which):

    # which = int(N_SAMPLES *0.98)
    track = shifted_movies[which][::, ::, ::, ::]
    track2 = autoencoder.predict(track[np.newaxis, ::, ::, ::, ::])
    track2 = track2[0][::, ::, ::, ::]

    # track2 = autoencoder.predict(track)
    # for j in range(16):
    #     new = new_pos[::, -1, ::, ::, ::]
    #     track = np.concatenate((track, new), axis=0)

    # And then compare the predictions
    # to the ground truth
    # track2 = shifted_movies[which][::, ::, ::, ::]

    for i in range(sequenceLength):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(131)
        # if i >= 7:
        # ax.text(1, 3, 'Ground Truth', fontsize=20, color='w')
        ax.text(2, 4, 'Ground Truth', fontsize=16, color='red')
        # else:
            # ax.text(1, 3, 'Inital trajectory', fontsize=20)
        toplot = track[i, ::, ::, ::]
        plt.imshow(toplot)

        ax = fig.add_subplot(132)
        ax.text(2, 4, 'Recovered', fontsize=16, color='red')
        toplot = track2[i, ::, ::, ::]
        plt.imshow(toplot)

        ax = fig.add_subplot(133)
        ax.text(2, 4, 'Recovered(Rescaled)', fontsize=16, color='red')
        toplot = toplot / np.amax(toplot)
        plt.imshow(toplot)

        plt.savefig('predict/%05i_animate.png' % (i + 1))

    images = []
    STR_PATH = "predict/"
    STR_FILE = ""
    STR_SUFFIX = "_animate.png"

    for i in range(sequenceLength):
        filename = STR_PATH+STR_FILE+ '%05i'%(i+1) +STR_SUFFIX
        im = imageio.imread(filename)
        images.append(im)

    imageio.mimsave('./result'+'_'+setup_name+'_'+str(num_epochs)+'.gif', images)

def main(num_epochs=EPOCH):

    ###################################
    # Model Define
    ###################################

    t = time.time()
    print("--- Start Assembling the Model ---")

    inputs = Input(shape=(sequenceLength,224,224,3))
    
    # conved = TimeDistributed(Lambda(MyCNN), input_shape=(sequenceLength,40,40,1)) (inputs)
 
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'), input_shape=(sequenceLength,224,224,3))(inputs)
    # x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Flatten())(x)

    x = Dense(3000, activation='relu')(x)
    # x = Reshape((15,10*10*4))(x)

    # x = LSTM(400, activation='tanh', return_sequences=True)(x)
    x = LSTM(3000, activation='tanh')(x)
    print(K.int_shape(x))

    x = RepeatVector(sequenceLength)(x)
    print(K.int_shape(x))
    print('--- Defining Decoder ---')

    x = LSTM(3000, activation='tanh', return_sequences=True)(x)
    print(K.int_shape(x))

    x = Dense(7*7*512, activation='relu')(x)
    # x = Reshape((15,10,10,4))(x)
    x = TimeDistributed(Reshape((7,7,512)))(x)
    
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    # deconved = TimeDistributed(Conv2D(1, (3, 3), padding='same', activation='relu'))(x)

    deconved = TimeDistributed(Conv2D(3, (3, 3), padding='same', activation='sigmoid'))(x)
    # deconved = TimeDistributed(Dense(3, activation='sigmoid'))(x)

    elapsed = time.time() - t
    print("%.2f seconds to define the model" % elapsed )
    t = time.time()

    autoencoder = Model(output=deconved,input=inputs)
    # myoptmizer = RMSprop(lr=0.1, decay=1e-4)
    # autoencoder.compile(loss='mean_squared_error', optimizer=myoptmizer)
    autoencoder.compile(loss='mean_squared_error', optimizer='RMSprop')

    plot_model(autoencoder, to_file='model.png', show_shapes=True)

    print('--- Finish Compile and Plot Model ---')

    ###################################
    # Training
    ###################################

    t = time.time()

    # Train the network
    if GENERATE_DATA:
        shifted_movies = generate_movies(n_samples=N_SAMPLES)
    else:
        noisy_movies, shifted_movies = get_ucf_data(n_samples=N_SAMPLES)

    elapsed = time.time() - t
    print("%.2f seconds to load the dataset" % elapsed )


    # seq.fit(noisy_movies[:1000], shifted_movies[:1000], 
    autoencoder.fit(shifted_movies[:int(N_SAMPLES*0.95)], shifted_movies[:int(N_SAMPLES*0.95)],
            batch_size=BATCHSIZE,
            epochs=num_epochs, 
            validation_split=0.05,
            callbacks=[TensorBoard(log_dir=LOG_DIR+'/convlstm_'+setup_name+'/epoch_'+str(num_epochs))])

    ###################################
    # Predicting
    ###################################

    # plot_val(int(N_SAMPLES*0.99))
    which = int(N_SAMPLES *0.98)
    track = shifted_movies[which][::, ::, ::, ::]
    track2 = autoencoder.predict(track[np.newaxis, ::, ::, ::, ::])
    track2 = track2[0][::, ::, ::, ::]

    for i in range(sequenceLength):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(131)
        ax.text(2, 4, 'Ground Truth', fontsize=16, color='red')
        toplot = track[i, ::, ::, ::]
        plt.imshow(toplot)

        ax = fig.add_subplot(132)
        ax.text(2, 4, 'Recovered', fontsize=16, color='red')
        toplot = track2[i, ::, ::, ::]
        plt.imshow(toplot)

        ax = fig.add_subplot(133)
        ax.text(2, 4, 'Recovered(Rescaled)', fontsize=16, color='red')
        toplot = toplot / np.amax(toplot)
        plt.imshow(toplot)

        plt.savefig('predict/%05i_animate.png' % (i + 1))

    images = []
    STR_PATH = "predict/"
    STR_FILE = ""
    STR_SUFFIX = "_animate.png"

    for i in range(sequenceLength):
        filename = STR_PATH+STR_FILE+ '%05i'%(i+1) +STR_SUFFIX
        im = imageio.imread(filename)
        images.append(im)

    imageio.mimsave('./result'+'_'+setup_name+'_'+str(num_epochs)+'.gif', images)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print ("Autoencoder for self-geneated moving squares movies:")
        print ("arg:\t[NUM_EPOCHS](500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)

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
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils.data_utils import get_file

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

import data_seq # Generating UCF sequences data

GENERATE_DATA = False
# 1 if generate aritificial data, 0 if use UCF101 data

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


weights_file = "/home/lab.analytics.northwestern.edu/yma/git/data/checkpoints/ucf_vgg16_simple_seq1_convlstm.014-0.04.hdf5"
# weights_file = ""
LOG_DIR = "../../tensorboard/log/"
data_path = "../../data/UCF/"
MAX_EPOCH = 150
sequenceLength = 1
setup_name = "ucf_vgg16_simple_seq1_convlstm"
UCF_CLASS_LIMIT = 1
BATCHSIZE = 5
MODE = "train"
STEPS_PER_EPOCH_TRAIN = 200


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

def generate_movies(n_samples=1000, n_frames=sequenceLength):
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
            # w = np.random.randint(30, 59)

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

def main(mode=MODE,num_epochs=MAX_EPOCH):

    ###################################
    # Model Define
    ###################################

    t = time.time()
    print("--- Start Assembling the Model ---")

    inputs = Input(shape=(sequenceLength,224,224,3))
    
    # conved = TimeDistributed(Lambda(MyCNN), input_shape=(sequenceLength,40,40,1)) (inputs)
 
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv1'), input_shape=(sequenceLength,224,224,3))(inputs)
    # x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block1_pool'))(x)
    # x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1'))(x)
    # x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block2_pool'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block3_pool'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block4_pool'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3'))(x)


    # LSTM part
    x = TimeDistributed(MaxPooling2D((2, 2), name='block5_pool'))(x)

    # x = TimeDistributed(Flatten())(x)

    x = ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True)(x)

    x = ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True)(x)
    
    # x = Dense(2000, activation='relu')(x)
    # x = Reshape((15,10*10*4))(x)

    # x = LSTM(400, activation='tanh', return_sequences=True)(x)
    # x = LSTM(2000, activation='tanh')(x)
    # print(K.int_shape(x))

    # x = RepeatVector(sequenceLength)(x)
    # print(K.int_shape(x))
    # print('--- Defining Decoder ---')

    # x = LSTM(2000, activation='tanh', return_sequences=True)(x)
    # print(K.int_shape(x))

    # x = Dense(7*7*512, activation='relu')(x)
    # x = Dense(56*56*64, activation='relu')(x)
    # x = Dense(28*28*256, activation='relu')(x)
    # x = Reshape((15,10,10,4))(x)
    # x = TimeDistributed(Reshape((7,7,512)))(x)
    # x = TimeDistributed(Reshape((56,56,64)))(x)
    # x = TimeDistributed(Reshape((28,28,256)))(x)
    
    x = TimeDistributed(UpSampling2D((2, 2)))(x)



    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    # # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    # x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2'))(x)
    # x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1'))(x)
    # x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)

    # x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2'))(x)
    # x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv1'))(x)
    # x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
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
        shifted_movies = generate_movies(n_samples=1000)
    else:
        data = data_seq.DataSet(seq_length=sequenceLength,class_limit=UCF_CLASS_LIMIT)
        ucf_train_generator =  data.seq_generator(BATCHSIZE, 'train', 'images')
        ucf_val_generator =  data.seq_generator(BATCHSIZE, 'test', 'images')

    # X, y  = next(ucf_train_generator)
    # print("maximum value in X is:%d"%X.max())

    elapsed = time.time() - t
    print("%.2f seconds to load the dataset" % elapsed )

    callbacks_func = [
        TensorBoard(log_dir=LOG_DIR+'/convlstm_'+setup_name+'/epoch_'+str(num_epochs)), 
        EarlyStopping(patience=5), 
        ModelCheckpoint(
            filepath='../../data/checkpoints/'+setup_name+'.{epoch:03d}-{val_loss:.2f}.hdf5',
            verbose=1,
            save_best_only=True)
        ]

    if weights_file is None or weights_file =="":
        # print("Random initilze the weights.")
        print("Loading pretrained VGG16 weights.")
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        autoencoder.load_weights(weights_path, by_name=True)

    else:
        print("Loading saved model: %s." % weights_file)
        autoencoder.load_weights(weights_file, by_name=True)

    if mode=="train":
        autoencoder.fit_generator(ucf_train_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,            
        epochs=num_epochs, 
        validation_data=ucf_val_generator,
        validation_steps=STEPS_PER_EPOCH_TRAIN*0.1,
        callbacks=callbacks_func 
        )
    elif mode =="inf":

        ###################################
        # Predicting
        ###################################

        # plot_val(int(N_SAMPLES*0.99))
        # which = int(N_SAMPLES *0.98)

        ucf_val_generator =  data.seq_generator(1, 'test', 'images')

        for index in range(10):
            print("Ploting %d gif for inference."%index)

            track,track_ = next(ucf_val_generator)
            track2 = autoencoder.predict(track)
            track = track[0][::, ::, ::, ::]
            # print(track.shape)
            # print(track2.shape)
            track2 = track2[0][::, ::, ::, ::]
            # print(track2.shape)


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
                maxvalue = np.amax(toplot)
                if maxvalue <= 0:
                    maxvalue = 1
                toplot = toplot / maxvalue
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

            imageio.mimsave('./result'+'_'+setup_name+'_'+str(index)+'.gif', images)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv) or ('help' in sys.argv):
        print ("Autoencoder for ucf101 dataset:")
        print ("arg:\t[train\inf(infernce)]")
        print ("arg:\t[NUM_EPOCHS](500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['mode'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)

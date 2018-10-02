# This is a sequential classification

""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras import backend as K
from keras.layers import Input

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

sys.path.append(os.path.expanduser('/home/lab.analytics.northwestern.edu/yma/git/videodl/seq_inquiry'))
import data_seq # Generating UCF sequences data
from utility import *
from mymodels import *
import conv_ae_config as config

# 1 if generate aritificial data, 0 if use UCF101 data

os.environ['CUDA_VISIBLE_DEVICES']=config.CUDA_VISIBLE_DEVICES

###################################
# Data Loading
###################################

# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

def main(mode=config.MODE,num_epochs=config.MAX_EPOCH):

    ###################################
    # Model Define
    ###################################

    t = time.time()
    print("--- Start Assembling the Model ---")

    inputs = Input(shape=(config.sequenceLength,224,224,3))
    
    encoder, autoencoder =  ConvAutoEncoder(inputs, sequenceLength = config.sequenceLength)
    
    elapsed = time.time() - t
    print("%.2f seconds to define the model" % elapsed )
    t = time.time()


    ###################################
    # Training
    ###################################

    t = time.time()

    # Train the network
    if config.GENERATE_DATA:
        shifted_movies = generate_movies(n_samples=1000, n_frames=config.sequenceLength)
    else:
        data = data_seq.DataSet(data_dir = config.data_path,
                                seq_length=config.sequenceLength,
                                class_limit=config.UCF_CLASS_LIMIT)
        
        ucf_train_generator =  data.seq_generator(config.BATCHSIZE, 'train', 'images')
        ucf_val_generator =  data.seq_generator(config.BATCHSIZE, 'test', 'images')

    # X, y  = next(ucf_train_generator)
    # print("maximum value in X is:%d"%X.max())

    elapsed = time.time() - t
    print("%.2f seconds to load the dataset" % elapsed )

    callbacks_func = [
        TensorBoard(log_dir=config.LOG_DIR+'/convlstm_'+config.setup_name+'/epoch_'+str(num_epochs)), 
        EarlyStopping(patience=30), 
        ModelCheckpoint(
            filepath='../../data/checkpoints/'+config.setup_name+'.{epoch:03d}-{val_loss:.4f}.hdf5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            # save_weights_only=True
            )
        ]

    if config.weights_file is None or config.weights_file =="":
        # print("Random initilze the weights.")
        print("Loading pretrained VGG16 weights.")
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    config.WEIGHTS_PATH,
                                    cache_subdir='models')
        autoencoder.load_weights(weights_path, by_name=True)

    else:
        print("Loading saved model: %s." % config.weights_file)
        autoencoder.load_weights(config.weights_file, by_name=True)

    if mode=="train":
        autoencoder.fit_generator(ucf_train_generator,
        steps_per_epoch=config.STEPS_PER_EPOCH_TRAIN,            
        epochs=num_epochs, 
        validation_data=ucf_val_generator,
        validation_steps=config.STEPS_PER_EPOCH_TRAIN*0.1,
        callbacks=callbacks_func 
        )
    elif mode =="inf":

        ###################################
        # Predicting
        ###################################

        # plot_val(int(N_SAMPLES*0.99))
        # which = int(N_SAMPLES *0.98)

        for inf_mode in ["train","test"]:


            ucf_val_generator =  data.seq_generator(1, inf_mode, 'images')

            for index in range(10):
                print("Ploting %d gif for inference."%index)

                track,track_ = next(ucf_val_generator)
                track2 = autoencoder.predict(track)
                track = track[0][::, ::, ::, ::]
                # print(track.shape)
                # print(track2.shape)
                track2 = track2[0][::, ::, ::, ::]
                # print(track2.shape)


                for i in range(config.sequenceLength):
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

                for i in range(config.sequenceLength):
                    filename = STR_PATH+STR_FILE+ '%05i'%(i+1) +STR_SUFFIX
                    im = imageio.imread(filename)
                    images.append(im)

                imageio.mimsave('result_'+inf_mode+'/'+setup_name+str(index)+'.gif', images)

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


# coding: utf-8

# The goal is to generate a UCF dataset with customized setting.  
# 
# The prefered target output is: ((n_samples, n_frames, row, col, 3), dtype=np.float)
# x_train
# y_train is the class of this 

# In[134]:

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

import csv
import os
import time
import sys
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import pylab as plt
from scipy.misc import toimage

import imageio
import cv2
import numpy as np
import numpngw
import skimage

from IPython.display import HTML


# In[21]:

print os.getcwd()


# In[3]:

GENERATE_DATA = 1 
LOG_DIR = "../../tensorboard/log/"
EPOCH = 150
sequenceLength = 3
setup_name = "clrmvsq_simple_vgg_a"
N_SAMPLES = 1000
BATCHSIZE = 5
ucf_generate_fps = 2  # The fps to sample from the original UCF data to generate the train and val set
data_path = "../../data/UCF/"

batch_size=20
data_type = 'images'
concat=False


# In[173]:

import data_seq
data_seq = reload(data_seq)


# In[174]:

data = data_seq.DataSet(seq_length=5,class_limit=1)


# In[175]:

print data.data[:1], '\n'
print data.classes[:5]
print data.image_shape


# In[176]:

generator = data.seq_generator(batch_size, 'train', 'images')


# In[184]:

X,y  = next(generator);


# In[185]:

print X.shape
# print X[0,0,::,::,0]


# In[186]:

# print X.shape
# print X[0,0,::,::,::]
# plt.imshow(X[0,0,::,::,::])
# plt.show()
# plt.imshow(X[0,10,::,::,::])
# plt.show()
# toimage(X[0,0,::,::,::]).show()


# In[187]:

images = X[0]
imageio.mimsave('./movie.gif', images)
HTML('<img src="./movie.gif">')


# In[188]:

images = X[1]
imageio.mimsave('./movie.gif', images)
HTML('<img src="./movie.gif">')


# In[ ]:




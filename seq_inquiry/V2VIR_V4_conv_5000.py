
# coding: utf-8

# # 20180930 Update
# 
# - reorg codes
# 	- flex
# 	- clearness
# 	- we should memic the strucutr of DiDi's OD flow prediction model code:
# 		- one function -> build model (different model strucutre)
# 		- one function -> train the model (specify small/large UCF or other dataset)
# 		- oen function -> inquiry on UCF
# 		- one function get accuracy description and save to txt
# - run experiments
# 	- DTW vs. simple matching (need 2 differnt length of video in inquiry set)
# 	- ConvLSTM vs. LSTM vs. Conv2D (sort of frame based VGG ) vs. (VidSig) (Non-DL method)
# 
# # A simple test for inquiry model
# 
# 
# Here is the setting:
# 
# - My thought: Yintai Ma [09:23] 
#   - Suppose I have a video X of 100 frames, denoting as X[0] to X[99]. We then split it to multiple clips without overlap. If then length of each clip is 5 frames, then we come up with a sequences of clips X[0:5],X[5:10],...,X[95:100]. We use the encoder to transfer the sequences of clips to sequences of embeddeds, reads y[0],y[1],...,y[20]. Now we randomly pick some frames from X as validation clip and transfer it into validation embedded y_hat. We want to see how y_hat is matching to the sequences of y[0]...y[20].
# 
# - Diego:
#   - You have video X. Randomly pick  sequences of frames (non overlapping). Say x[4:10], x[34:37],x[85:95]. Now concatenate them into a single video. This is now your query. From here, create embedding sequence y[0],…y[K]. Now do DTW of y against your encoded sequences of the videos in the database.
#   - This is subject to experimentation. I agree that there should be overlap. Overlap by half.
# 
# - My thought: There are many variations for the implementation:
#   1. we keep overlaps for y. We transfer X[0:5],X[1:6],....X[94:99],X[95:100] into y[0]...y[100]. Now we want to see how y_hat is matching to y[].
#   2. When we pick some frames from X to composite validation clips, do we always pick 5 consecutive frames? Should we ever transfer X[0]+X[2:6] into y_hat?
#   3. How we match y_hat to y[] if y_hat is also a sequences? I think this is where DTW comes in right? If y_hat is just one embedded, then what we need is basically a simple comparison between the distance of two embedded. However, if y_hat is a sequences, say it has y_hat[0] and y_hat[1], then we will need to use DTW to consider the case where both y_hat[0] matches to y[0] and y_hat[1] matches to y[3] are the best query retrieve.
# 

# In[1]:


import sys,os,os.path
sys.path.append(os.path.expanduser('/home/lab.analytics.northwestern.edu/yma/git/videodl/seq_inquiry'))
os.environ['CUDA_VISIBLE_DEVICES']='2'

from keras.models import load_model

import csv
import os
import time
import sys
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import pylab as plt
# from scipy.misc import toimage
import pickle

import imageio
import cv2
import numpy as np
import numpngw
import pandas as pd 

from IPython.display import HTML
import random

from mymodels import *
from Video2videoInquiry import *

import data_seq


# # Load trained model

# In[3]:


LOAD_DATABASE_FROM_PKL = False

# import conv_ae_config as config
# model_file = "ucf_vgg16_seq3_convlstm.001-0.0689.hdf5" # encoding filter = 8
# encoder, autoencoder =  ConvAutoEncoder(sequenceLength = config.sequenceLength)
# database_file = "/scratch/yma/data/inq_encoded_class100_video500_conv_convlstm_8.pkl"

# import simple_convlstm_ae_config as config
# model_file_simple_convlstm = "ucf_seq3_simple_convlstm.003-0.0284.hdf5" # encoding filter = 8
# encoder_simple_convlstm, autoencoder_simple_convlstm =  SimpleConvLstmAutoEncoder(sequenceLength = config.sequenceLength)
# database_file_simple_convlstm = "/scratch/yma/data/inq_encoded_class100_video500_simple_convlstm_8.pkl"

import simple_conv_ae_config as config
# model_file = "ucf_seq3_simple_conv.001-0.0231.hdf5" # encoding filter = 8
model_file_simple_conv = "ucf_seq3_simple_conv.016-0.0112.hdf5" # encoding filter = 8
encoder_simple_conv, autoencoder_simple_conv =  SimpleConvAutoEncoder(sequenceLength = config.sequenceLength)
database_file_simple_conv = "/scratch/yma/data/inq_encoded_class100_video5000_simple_conv_8.pkl"

# import simple_lstm_ae_config as config
# model_file_simple_lstm = "ucf_seq3_simple_lstm.002-0.0726.hdf5" # encoding filter = 8
# encoder_simple_lstm, autoencoder_simple_lstm =  SimpleLstmAutoEncoder(sequenceLength = config.sequenceLength)
# database_file_simple_lstm = "/scratch/yma/data/inq_encoded_class100_video500_simple_lstm_400.pkl"

models = [
# {
# "setup":"simple_lstm",
# "model_file":model_file_simple_lstm, 
# "encoder":encoder_simple_lstm,
# "database_file":database_file_simple_lstm},
{
"setup":"simple_conv",
"model_file":model_file_simple_conv, 
"encoder":encoder_simple_conv,
"database_file":database_file_simple_conv},
# {"setup":"simple_convlstm",
# "model_file":model_file_simple_convlstm, 
# "encoder":encoder_simple_convlstm,
# "database_file":database_file_simple_convlstm},
]

model_dir = "/home/lab.analytics.northwestern.edu/yma/git/data/checkpoints/"
data_path = "/scratch/yma/git/five-video-classification-methods/data"

for model_dict in models:
    print("Loading weights for setup:", model_dict["setup"])
    print("The output dimension of encoder is:", model_dict["encoder"].output.shape)
    print(model_dict["encoder"].summary())
    model_file = os.path.join(model_dir, model_dict["model_file"])
    model_dict["encoder"].load_weights(model_file, by_name=True)

    

SEQ_LENGTH = config.sequenceLength
# the number of frames in each clip

# N_database = 500
# N_database = 5
# N_database = 50

inq_length = 4
# the number of clips in the inquiry, no overlap

DATASET_CLASS_LIMIT = 100
# number of class is the dataset

DATASET_VIDEO_IN_CLASS_LIMIT = 50

FLAG_RANDOM_CLASS = True
# whether randomly pick classes in the dataset


# In[ ]:


for model_dict in models:
    encoder = model_dict["encoder"]
    database_file = model_dict["database_file"]

    data = initilize(encoder, data_path = data_path, 
                     seq_length = config.sequenceLength, 
                     class_limit = DATASET_CLASS_LIMIT, 
                     num_video_in_each_class = DATASET_VIDEO_IN_CLASS_LIMIT,
                     random_class=FLAG_RANDOM_CLASS)

    if LOAD_DATABASE_FROM_PKL:
        with open(database_file, 'rb') as f:
            database = pickle.load(f)
    else:
        database = create_database(encoder, data, 
                                   seq_length = config.sequenceLength, 
                                   class_limit = DATASET_CLASS_LIMIT, 
                                   num_video_in_each_class = DATASET_VIDEO_IN_CLASS_LIMIT)
        # DATATYPE EXAMPLE: database.append((seqY, smp[2]))
        with open(database_file, 'wb') as f:
            pickle.dump(database, f)


# In[ ]:


# inqs = get_inquiry(data, if_random = True)
# seqY = inqs[1]

inq_dict, inq_result = inquiry_in_database(data, database, config, inq_length = inq_length , match_method = "dtw")
show_inquriy_stats(inqs, inq_result, show_top_limit = 5)

print("*"*20)
print("*"*20)
print("*"*20)

inq_dict, inq_result = inquiry_in_database(data, database, config, inq_length = inq_length, match_method = "naive", inq_dict = inq_dict)
show_inquriy_stats(inqs, inq_result, show_top_limit = 5)


# In[ ]:


index = np.random.choice(len(database))
inquiry_seqY = database[index][0]
inqs = database[index]
        


# In[2]:


data = data_seq.DataSet(data_dir = data_path, seq_length=3,class_limit=100, random_class=True)
index = np.random.choice(len(data.data))
smp = data.data[index]


# In[3]:


smp


# In[ ]:


# (top_cat_same,top_cat_same_hit, Nth_score_avg, Hit_itself_avg) = multiple_test(data, run_times=100, if_itself=False)

res_dict = multiple_test(data, run_times=5, if_itself=False)


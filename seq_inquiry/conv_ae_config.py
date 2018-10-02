
GENERATE_DATA = False

CUDA_VISIBLE_DEVICES = "2"

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


# weights_file = "/home/lab.analytics.northwestern.edu/yma/git/data/checkpoints/ucf_vgg16_seq3_convlstm.040-0.0857.hdf5"
# weights_file = "/home/lab.analytics.northwestern.edu/yma/git/data/checkpoints/ucf_vgg16_seq6_convlstm.040-0.0842.hdf5"
weights_file = ""


LOG_DIR = "../../tensorboard/log/"
# data_path = "../../data/UCF/"
data_path = "/scratch/yma/git/five-video-classification-methods/data"
MAX_EPOCH = 150
sequenceLength = 3
setup_name = "ucf_vgg16_seq3_convlstm"
UCF_CLASS_LIMIT = 1
BATCHSIZE = 5
MODE = "train"
STEPS_PER_EPOCH_TRAIN = 1000
LSTM_STATE = 7*7*64
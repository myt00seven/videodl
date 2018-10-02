from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LSTM, RepeatVector
from keras.layers.wrappers import *
from keras.layers.core import *
from keras.layers import Conv3D
from keras.layers import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils.data_utils import get_file

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

def ConvAutoEncoder(inputs, sequenceLength):
    
    # conved = TimeDistributed(Lambda(MyCNN), input_shape=(sequenceLength,40,40,1)) (inputs)
 
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv1'), input_shape=(sequenceLength,224,224,3))(inputs)
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block1_pool'))(x)
    # x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block2_pool'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block3_pool'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block4_pool'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)


    # LSTM part
    x = TimeDistributed(MaxPooling2D((2, 2), name='block5_pool'))(x)

    # x = TimeDistributed(Flatten())(x)

    x = ConvLSTM2D(filters=64, kernel_size=(3, 3),padding='same', return_sequences=True)(x)
    
    # x = Dense(2000, activation='relu')(x)
    # x = Reshape((15,10*10*4))(x)

    # x = LSTM(400, activation='tanh', return_sequences=True)(x)
    # x = LSTM(2000, activation='tanh')(x)
    # x = LSTM(LSTM_STATE, activation='tanh',return_sequences=True)(x)
    # print(K.int_shape(x))

    # x = RepeatVector(sequenceLength)(x)
    # print(K.int_shape(x))
    
    encoded = x
    encoder = Model(output=encoded,input=inputs)
    encoder.compile(loss='mean_squared_error', optimizer='RMSprop')
    
    print('--- Defining Decoder ---')

    x = ConvLSTM2D(filters=64, kernel_size=(3, 3),padding='same', return_sequences=True)(x)


    # x = LSTM(LSTM_STATE, activation='tanh', return_sequences=True)(x)
    # print(K.int_shape(x))

    # x = Dense(sequenceLength*7*7*64, activation='relu')(x)
    # x = Dense(56*56*64, activation='relu')(x)
    # x = Dense(28*28*256, activation='relu')(x)
    # x = Reshape((15,10,10,4))(x)
    # x = Reshape((sequenceLength,7,7,64))(x)
    # print(K.int_shape(x))

    # x = TimeDistributed(Reshape((7,7,64)))(x)
    # x = TimeDistributed(Reshape((56,56,64)))(x)
    # x = TimeDistributed(Reshape((28,28,256)))(x)
    
    x = TimeDistributed(UpSampling2D((2, 2)))(x)



    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2'))(x)
    # x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    # x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2'))(x)
    # x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
    # x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)

    # x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2'))(x)
    # x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv1'))(x)
    # x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    # deconved = TimeDistributed(Conv2D(1, (3, 3), padding='same', activation='relu'))(x)
    deconved = TimeDistributed(Conv2D(3, (3, 3), padding='same', activation='sigmoid'))(x)
    # deconved = TimeDistributed(Dense(3, activation='sigmoid'))(x)
    
    autoencoder = Model(output=deconved,input=inputs)
    # myoptmizer = RMSprop(lr=0.1, decay=1e-4)
    # autoencoder.compile(loss='mean_squared_error', optimizer=myoptmizer)
    autoencoder.compile(loss='mean_squared_error', optimizer='RMSprop')

    plot_model(autoencoder, to_file='model.png', show_shapes=True)

    print('--- Finish Compile and Plot Model ---')

    
    return encoder, autoencoder

######################################################
######################################################
######################################################
# ConvLstm Backup
######################################################
######################################################
######################################################

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
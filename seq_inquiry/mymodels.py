from keras.models import Model
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

# Num_Encoded_Filter = 64
Num_Encoded_Filter = 8

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

def ConvAutoEncoder(sequenceLength):
    
    inputs = Input(shape=(sequenceLength,224,224,3))
    
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
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='block5_conv3'))(x)


    # LSTM part
    x = TimeDistributed(MaxPooling2D((2, 2), name='block5_pool'))(x)

    # x = TimeDistributed(Flatten())(x)

    x = ConvLSTM2D(filters=Num_Encoded_Filter, kernel_size=(3, 3),padding='same', name="convlstm_before_encoded", return_sequences=True)(x)
    
    # x = Dense(2000, activation='relu')(x)
    # x = Reshape((15,10*10*4))(x)

    # x = LSTM(400, activation='tanh', return_sequences=True)(x)
    # x = LSTM(2000, activation='tanh')(x)
    # x = LSTM(LSTM_STATE, activation='tanh',return_sequences=True)(x)
    # print(K.int_shape(x))

    # x = RepeatVector(sequenceLength)(x)
    # print(K.int_shape(x))
    
    encoded = x
    print("Encoded embedding size: ", encoded.shape)
    encoder = Model(output=encoded,input=inputs)
    encoder.compile(loss='mean_squared_error', optimizer='RMSprop')
    
    print('--- Defining Decoder ---')

    x = ConvLSTM2D(filters=Num_Encoded_Filter, kernel_size=(3, 3),padding='same', return_sequences=True)(x)


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

def SimpleConvLstmAutoEncoder(sequenceLength):
    
    inputs = Input(shape=(sequenceLength,224,224,3))
    filters = [64,128,256,512,64]
    filters = [i/2 for i in filters]    
    
    # conved = TimeDistributed(Lambda(MyCNN), input_shape=(sequenceLength,40,40,1)) (inputs)
    
    x = ConvLSTM2D(filters=filters[0], kernel_size=(3, 3), input_shape=(None, 224, 224, 3),
                   padding='same', activation='relu', name = "Convlstm_1", return_sequences=True)(inputs)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block1_pool'))(x)
    x = BatchNormalization(name = "bn_1")(x)
    x = ConvLSTM2D(filters=filters[1], kernel_size=(3, 3), padding='same', activation='relu', 
                   name = "Convlstm_2", return_sequences=True)(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block2_pool'))(x)
    x = BatchNormalization(name = "bn_2")(x)
    x = ConvLSTM2D(filters=filters[2], kernel_size=(3, 3), padding='same', activation='relu', 
                   name = "Convlstm_3", return_sequences=True)(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block3_pool'))(x)
    x = BatchNormalization(name = "bn_3")(x)
    x = ConvLSTM2D(filters=filters[3], kernel_size=(3, 3), padding='same', activation='relu', 
                   name = "Convlstm_4", return_sequences=True)(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block4_pool'))(x)
    x = BatchNormalization(name = "bn_4")(x)
    x = ConvLSTM2D(filters=filters[-1], kernel_size=(3, 3), padding='same', activation='relu', 
                   name = "Convlstm_5", return_sequences=True)(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block5_pool'))(x)
    x = BatchNormalization(name = "bn_5")(x)

    x = ConvLSTM2D(filters=Num_Encoded_Filter, kernel_size=(3, 3),padding='same', name="convlstm_before_encoded", return_sequences=True)(x)
    
    encoded = x
    print("Encoded embedding size: ", encoded.shape)
    encoder = Model(output=encoded,input=inputs)
    encoder.compile(loss='mean_squared_error', optimizer='RMSprop')
    print('--- Defining Decoder ---')

    x = ConvLSTM2D(filters=Num_Encoded_Filter, kernel_size=(3, 3),padding='same', return_sequences=True)(x)
    
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = ConvLSTM2D(filters=filters[-1], kernel_size=(3, 3), padding='same', activation='relu', 
                   name = "decode_Convlstm_5", return_sequences=True)(x)
    x = BatchNormalization(name = "decode_bn_5")(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = ConvLSTM2D(filters=filters[3], kernel_size=(3, 3), padding='same', activation='relu', 
                   name = "decode_Convlstm_4", return_sequences=True)(x)
    x = BatchNormalization(name = "decode_bn_4")(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = ConvLSTM2D(filters=filters[2], kernel_size=(3, 3), padding='same', activation='relu', 
                   name = "decode_Convlstm_3", return_sequences=True)(x)
    x = BatchNormalization(name = "decode_bn_3")(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = ConvLSTM2D(filters=filters[1], kernel_size=(3, 3), padding='same', activation='relu', 
                   name = "decode_Convlstm_2", return_sequences=True)(x)
    x = BatchNormalization(name = "decode_bn_2")(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = ConvLSTM2D(filters=filters[0], kernel_size=(3, 3), padding='same', activation='relu', 
                   name = "decode_Convlstm_1", return_sequences=True)(x)
    x = BatchNormalization(name = "decode_bn_1")(x)
    
    deconved = TimeDistributed(Conv2D(3, (3, 3), padding='same', activation='sigmoid'))(x)
    
    autoencoder = Model(output=deconved,input=inputs)
    autoencoder.compile(loss='mean_squared_error', optimizer='RMSprop')

    plot_model(autoencoder, to_file='model_simple_convlstm.png', show_shapes=True)

    print('--- Finish Compile and Plot Model ---')
    
    return encoder, autoencoder

def SimpleConvAutoEncoder(sequenceLength):
    
    inputs = Input(shape=(sequenceLength,224,224,3))
    filters = [64,128,256,512,64]
    
        
    x = TimeDistributed(Conv2D(filters[0], (3, 3), padding='same', activation='relu', name='block1_conv1'), input_shape=(sequenceLength,224,224,3))(inputs)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block1_pool'))(x)
    x = TimeDistributed(Conv2D(filters[1], (3, 3), padding='same', activation='relu', name='block2_conv1'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block2_pool'))(x)
    x = TimeDistributed(Conv2D(filters[2], (3, 3), padding='same', activation='relu', name='block3_conv1'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block3_pool'))(x)
    x = TimeDistributed(Conv2D(filters[3], (3, 3), padding='same', activation='relu', name='block4_conv1'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block4_pool'))(x)
    x = TimeDistributed(Conv2D(filters[-1], (3, 3), padding='same', activation='relu', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), name='block5_pool'))(x)

    x = ConvLSTM2D(filters=Num_Encoded_Filter, kernel_size=(3, 3),padding='same', name="convlstm_before_encoded", return_sequences=True)(x)
    
    encoded = x
    print("Encoded embedding size: ", encoded.shape)
    encoder = Model(output=encoded,input=inputs)
    encoder.compile(loss='mean_squared_error', optimizer='RMSprop')
    
    print('--- Defining Decoder ---')

    x = ConvLSTM2D(filters=Num_Encoded_Filter, kernel_size=(3, 3),padding='same', 
                   name = "convlstm_decode",
                   return_sequences=True)(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(filters[-1], (3, 3), padding='same', activation='relu', name = "de_block5_conv2d"))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(filters[3], (3, 3), padding='same', activation='relu', name = "de_block4_conv2d"))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(filters[2], (3, 3), padding='same', activation='relu', name = "de_block3_conv2d"))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(filters[1], (3, 3), padding='same', activation='relu', name = "de_block2_conv2d"))(x)
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(filters[0], (3, 3), padding='same', activation='relu', name = "de_block1_conv2d"))(x)
    
    deconved = TimeDistributed(Conv2D(3, (3, 3), padding='same', activation='sigmoid'))(x)
    
    autoencoder = Model(output=deconved,input=inputs)
    autoencoder.compile(loss='mean_squared_error', optimizer='RMSprop')

    plot_model(encoder, to_file='encoder_simple_conv.png', show_shapes=True)
    plot_model(autoencoder, to_file='model_simple_conv.png', show_shapes=True)

    print('--- Finish Compile and Plot Model ---')
    
    return encoder, autoencoder

def SimpleLstmAutoEncoder(sequenceLength):
    
    inputs = Input(shape=(sequenceLength,224,224,3))
#     shapeSize =  sequenceLength
    shapeSize = 224 * 224 * 3
    filters = [400, 100, 400]
    
    x = Reshape((sequenceLength,shapeSize))(inputs)
    x = Dense(filters[0], activation='relu')(x)
    # x = Reshape((15,10*10*4))(x)

    x = LSTM(filters[0], activation='tanh', return_sequences=True)(x)
    x = BatchNormalization(name = "bn_1")(x)
    x = LSTM(filters[1], activation='tanh', return_sequences=True)(x)
    x = BatchNormalization(name = "bn_2")(x)
    x = LSTM(filters[-1], activation='tanh', return_sequences=True)(x)
    x = BatchNormalization(name = "bn_3")(x)
    
    encoded = x
    print("Encoded embedding size: ", encoded.shape)
    encoder = Model(output=encoded,input=inputs)
    encoder.compile(loss='mean_squared_error', optimizer='RMSprop')
    print('--- Defining Decoder ---')

    x = LSTM(filters[-1], activation='tanh', return_sequences=True)(x)
    x = BatchNormalization(name = "decoded_bn_3")(x)
    x = LSTM(filters[1], activation='tanh', return_sequences=True)(x)
    x = BatchNormalization(name = "decoded_bn_2")(x)
    x = LSTM(filters[0], activation='tanh', return_sequences=True)(x)
    x = BatchNormalization(name = "decoded_bn_1")(x)
    
    x = Dense(shapeSize, activation='relu')(x)
    x = Reshape((sequenceLength,224,224,3))(x)
    
    deconved = TimeDistributed(Conv2D(3, (3, 3), padding='same', activation='sigmoid'))(x)
    
    autoencoder = Model(output=deconved,input=inputs)
    autoencoder.compile(loss='mean_squared_error', optimizer='RMSprop')

    plot_model(autoencoder, to_file='model_simple_lstm.png', show_shapes=True)

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
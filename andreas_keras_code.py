### Define Encoder
inputs = [Input(shape=(sequenceLength,224,224,3), dtype='float32',name='main_input%d' % i) for i in range(0,numberCrops)]
encodes = [sharedVGG(i) for i in inputs]
merged_vector = merge(encodes,mode = 'ave')
flat = TimeDistributed(Flatten())(merged_vector)
encoder = LSTM(units=3136, return_sequences=True)(flat)
encoderModel = Model(input=inputs,output=encoder)
print('--- Defining Decoder ---')
x = LSTM(units = 3136,  return_sequences=True)(encoderModel.output)
x = Reshape((sequenceLength,7,7,64))(x)

# the size of input and out of LSTM can be different
# In Andreea's real code, she choose them to be the same.
# number of paramets in LSTM: length of seq * ((intput+1)(output)+output^2)


################  This part defines the Reverse #################

x = TimeDistributed(ReverseAvg(num_output_tensors= 6))(x)  #output= (None,6, 512, 7, 7)

x_i = [TimeDistributed(Lambda(splitTensor1, output_shape=splitTensor_output_shape))(x) ,TimeDistributed(Lambda(splitTensor2, output_shape=splitTensor_output_shape))(x),
   TimeDistributed(Lambda(splitTensor3, output_shape=splitTensor_output_shape))(x),TimeDistributed(Lambda(splitTensor4, output_shape=splitTensor_output_shape))(x),
   TimeDistributed(Lambda(splitTensor5, output_shape=splitTensor_output_shape))(x),TimeDistributed(Lambda(splitTensor6, output_shape=splitTensor_output_shape))(x)]

decodes = [sharedDecVGG(j) for j in x_i]

autoencoder = Model(input=inputs,output=decodes)

# [2:10] 
# VGG and DecVGG i have in separate models and i just called them here

######################## DEFINE MODEL #####################################
#Load VGG
shared_input = Input(shape=(sequenceLength,224,224,3), dtype='float32', name='shared_input')
shared_output = VGG(shared_input)
sharedVGG = Model(shared_input, shared_output)
#Load Decoder VGG
shared_input_dec = Input(shape=(sequenceLength, 7,7,64), dtype='float32', name='shared_input_dec')
shared_output_dec = VGGDec(shared_input_dec)
sharedDecVGG = Model(shared_input_dec, shared_output_dec)

### Define rest of the layers
inputs = [Input(shape=(sequenceLength,224,224,3), dtype='float32',name='main_input%d' % i) for i in range(0,numberCrops)]
encodes = [sharedVGG(i) for i in inputs]
merged_vector = merge(encodes,mode = 'ave')
flat = TimeDistributed(Flatten())(merged_vector)
encoder = LSTM(units=3136, return_sequences=True)(flat)
encoderModel = Model(input=inputs,output=encoder)
print('--- Defining Decoder ---')
x = LSTM(units = 3136,  return_sequences=True)(encoderModel.output)
x = Reshape((sequenceLength,7,7,64))(x)
x = TimeDistributed(ReverseAvg(num_output_tensors= 6))(x)  #output= (None,6, 512, 7, 7)
x_i = [TimeDistributed(Lambda(splitTensor1, output_shape=splitTensor_output_shape))(x) ,TimeDistributed(Lambda(splitTensor2, output_shape=splitTensor_output_shape))(x),
   TimeDistributed(Lambda(splitTensor3, output_shape=splitTensor_output_shape))(x),TimeDistributed(Lambda(splitTensor4, output_shape=splitTensor_output_shape))(x),
   TimeDistributed(Lambda(splitTensor5, output_shape=splitTensor_output_shape))(x),TimeDistributed(Lambda(splitTensor6, output_shape=splitTensor_output_shape))(x)]

decodes = [sharedDecVGG(j) for j in x_i]
autoencoder = Model(input=inputs,output=decodes)

# this is the complete model

def VGGDec(encoded):
   
   # Block 5 Decoder
   x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(encoded)
   x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
   x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
   
   # Block 4 Decoder
   x = UpSampling2D((2,2))(x)
   x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
   x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
   x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
   
   # Block 3 Decoder
   x = UpSampling2D((2,2))(x)
   x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
   x = Convolution2D(256, 3, 3, activation='relu',border_mode='same')(x)
   x = Convolution2D(256, 3, 3, activation='relu',border_mode='same')(x)
   
   # Block 2 Decoder
   x = UpSampling2D((2,2))(x)
   x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
   x = Convolution2D(128, 3, 3, activation='relu',border_mode='same')(x)
   
   # Block 1 Decoder
   x = UpSampling2D((2,2))(x)
   x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
   x = Convolution2D(64, 3, 3, activation='relu',border_mode='same')(x)

   
   x = UpSampling2D((2,2))(x)
   ### add extra conv layer
   decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
   
   return decoded

# [2:27] 
def VGG(main_input):
   
   #Block 1
   x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(main_input)
   #x= Convolution2D(64, 3, 3, activation='relu',border_mode='same')(x)
   x= MaxPooling2D(pool_size=(2, 2),border_mode='same')(x)
   
   # Block 2
   x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
   #x= Convolution2D(128, 3, 3, activation='relu',border_mode='same')(x)
   x= MaxPooling2D(pool_size=(2, 2),border_mode='same')(x)
   
   # Block 3
   x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
   #x = Convolution2D(256, 3, 3, activation='relu',border_mode='same')(x)
   #x = Convolution2D(256, 3, 3, activation='relu',border_mode='same')(x)
   x = MaxPooling2D(pool_size=(2, 2),border_mode='same')(x)
   
   # Block 4
   x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
   #x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
   #x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
   x = MaxPooling2D(pool_size=(2, 2),border_mode='same')(x)
   
   # Block 5
   x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
   #x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)
   #x = Convolution2D(512, 3, 3, activation='relu',border_mode='same')(x)

   encoded = MaxPooling2D(pool_size=(2, 2),border_mode='same')(x)  # 512*7*7

   # The output dimension is 512*7*7
   
   return encoded

   # ignore the commented layers. i did this to run a quicker version
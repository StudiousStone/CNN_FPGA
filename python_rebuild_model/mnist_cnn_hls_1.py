'''
This version adds some optimization regarding:
1.re-arrange the loop sturucture.
2.adds loop tiling(not done yet).
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

batch_size = 128
nb_classes = 10
nb_epoch = 2

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


model.load_weights('./mnist_cnn_trained.h5')

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


#read a number from a txt file
x_in = np.loadtxt('./3_number.txt')
#normalize
x_in = x_in/255
x_in = x_in.astype('float32')


#then feed it to the trained model, get the output of first layer
x_in =  x_in.reshape(1,28,28,1)
print('the input image shape: ',x_in.shape)
#input of one picture to our own cnn
#im_in = x_in.reshape(28,28,1)
im_in = x_in.reshape(784) #flatten the input image
np.savetxt('./input_3.txt',im_in,fmt='%5.20f')
#layer_name:
#layer 0: convolution2d_1
#layer 1: activation_1
#layer 2: convolution2d_2
#layer 3: activation_2
#layer 4: maxpooling2d_1
#layer 5: dropout_1
#layer 6: flatten_1
#layer 7: dense_1
#layer 8: activation_3
#layer 9: dropout_2
#layer 10: dense_2
#layer 11: activation_4
w_0 = model.layers[0].get_weights() # weights and biases of layer 0
print('shape of layer 0 weighs(conv1):',w_0[0].shape) # weight of layer 0
print('shape of layer 0 biases(conv1):',w_0[1].shape) # bias of layer 0
w_2 = model.layers[2].get_weights() # weights and biases of layer 2
print('shape of layer 2 weighs(conv2):',w_2[0].shape) # weight of layer 2
print('shape of layer 2 biases(conv2):',w_2[1].shape) # bias of layer 2
w_7 = model.layers[7].get_weights() # weights and biases of layer 7
print('shape of layer 7 weighs(dense1):',w_7[0].shape) # weight of layer 7
print('shape of layer 7 biases(dense1):',w_7[1].shape) # bias of layer 7
w_10 = model.layers[10].get_weights() # weights and biases of layer 10
print('shape of layer 10 weighs(dense2):',w_10[0].shape) # weight of layer 10
print('shape of layer 10 biases(dense2):',w_10[1].shape) # bias of layer 10

get_1st_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[0].output])

get_2nd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[1].output])

get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[2].output])

get_4th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

get_5th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[4].output])

get_8th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[7].output])

get_11th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[10].output])

get_12th_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[11].output])
#put the data into the model with test mode
layer_out_0 = get_1st_layer_output([x_in,0])[0]
layer_out_1 = get_2nd_layer_output([x_in,0])[0]
layer_out_2 = get_3rd_layer_output([x_in,0])[0]
layer_out_3 = get_4th_layer_output([x_in,0])[0]
layer_out_4 = get_5th_layer_output([x_in,0])[0]
layer_out_7 = get_8th_layer_output([x_in,0])[0]
layer_out_10 = get_11th_layer_output([x_in,0])[0]
layer_out_11 = get_12th_layer_output([x_in,0])[0]
layer_out_0=layer_out_0.reshape(26,26,32)
layer_out_1=layer_out_1.reshape(26,26,32)
layer_out_2=layer_out_2.reshape(24,24,32)
layer_out_3=layer_out_3.reshape(24,24,32)
layer_out_4=layer_out_4.reshape(12,12,32)
layer_out_7=layer_out_7.reshape(128)
layer_out_10=layer_out_10.reshape(10)
layer_out_11=layer_out_11.reshape(10)
print('the shape of 1st layer output(pre-trained model): ',layer_out_0.shape)
print('the shape of 2nd layer output(pre-trained model): ',layer_out_1.shape)
print('the shape of 3rd layer output(pre-trained model): ',layer_out_2.shape)
print('the shape of 4th layer output(pre-trained model): ',layer_out_3.shape)
print('the shape of 5th layer output(pre-trained model): ',layer_out_4.shape)
print('the shape of 8th layer output(pre-trained model): ',layer_out_7.shape)
print('the shape of 11th layer output(pre-trained model): ',layer_out_10.shape)
print('the shape of 12th layer output(pre-trained model): ',layer_out_11.shape)


#read the weights and biases, save them to txt files
weights_0 = w_0[0] #the weights of first convolutional layer
weights_0 = weights_0.reshape(288) #flatten the weights
np.savetxt('./mnist_weight_bias/weights_0.txt',weights_0,fmt='%5.20f')
biases_0 = w_0[1]  #the biases of first convolutional layer
np.savetxt('./mnist_weight_bias/biases_0.txt',biases_0,fmt='%5.20f')

weights_2 = w_2[0] #the weights of second convolutional layer
weights_2 = weights_2.reshape(9216)
np.savetxt('./mnist_weight_bias/weights_2.txt',weights_2,fmt='%5.20f')
biases_2 = w_2[1] #the biases of second convolutional layer
np.savetxt('./mnist_weight_bias/biases_2.txt',biases_2,fmt='%5.20f')

weights_7 = w_7[0] #the weights of second convolutional layer
weights_7 = weights_7.reshape(589824)
np.savetxt('./mnist_weight_bias/weights_7.txt',weights_7,fmt='%5.20f')
biases_7 = w_7[1] #the biases of second convolutional layer
np.savetxt('./mnist_weight_bias/biases_7.txt',biases_7,fmt='%5.20f')

weights_10 = w_10[0] #the weights of second convolutional layer
weights_10 = weights_10.reshape(1280)
np.savetxt('./mnist_weight_bias/weights_10.txt',weights_10,fmt='%5.20f')
biases_10 = w_10[1] #the biases of second convolutional layer
np.savetxt('./mnist_weight_bias/biases_10.txt',biases_10,fmt='%5.20f')

#let's re-build the CNN here
#first layer, convolution 1
stride = 1
layer_out_0_fm = np.zeros((26,26,32),dtype=np.float32) #initialize output
for out_dep in range(32): #output put depth, number of filters
    for row in range(26): #output row
        for col in range(26): #output column
            layer_out_0_fm[row][col][out_dep] += biases_0[out_dep]


for i in range(3): #filter size
    for j in range(3): #fileter size
        for row in range(26): #output row
            for col in range(26): #output column
                for out_dep in range(32): #output put depth, number of filters
                    for in_dep in range(1): #input feature maps
                        layer_out_0_fm[row][col][out_dep] += weights_0[((i*3+j)*1+in_dep)*32+out_dep]*im_in[((stride*row+i)*28+(stride*col+j))*1+in_dep]

#second layer, activation 1
layer_out_1_fm = np.zeros((26,26,32),dtype=np.float32) #initialize output
for out_dep in range(32): #output put depth, number of filters
    for row in range(26): #output row
        for col in range(26): #output column
            if layer_out_0_fm[row][col][out_dep] < 0:
                layer_out_1_fm[row][col][out_dep] = 0
                #print('first layer output: ',layer_out_0_fm[row][col][out_dep])
                #print('activation output: ',layer_out_1_fm[row][col][out_dep])
            else:
                layer_out_1_fm[row][col][out_dep] = layer_out_0_fm[row][col][out_dep]
                #print('first layer output: ',layer_out_0_fm[row][col][out_dep])
                #print('activation output: ',layer_out_1_fm[row][col][out_dep])

#third layer, convolution 2
layer_out_2_fm = np.zeros((24,24,32),dtype=np.float32) #initialize output
for out_dep in range(32): #output put depth, number of filters
    for row in range(24): #output row
        for col in range(24): #output column
            layer_out_2_fm[row][col][out_dep] += biases_2[out_dep]

for i in range(3): #filter size
    for j in range(3): #fileter size
        for row in range(24): #output row
            for col in range(24): #output column
                for out_dep in range(32): #output put depth, number of filters
                    for in_dep in range(32): #input feature maps
                        layer_out_2_fm[row][col][out_dep] += weights_2[((i*3+j)*32+in_dep)*32+out_dep]*layer_out_1_fm[stride*row+i][stride*col+j][in_dep]
                        
#fourth layer, activation 2
layer_out_3_fm = np.zeros((24,24,32),dtype=np.float32) #initialize output
for out_dep in range(32): #output put depth, number of filters
    for row in range(24): #output row
        for col in range(24): #output column
            if layer_out_2_fm[row][col][out_dep] < 0:
                layer_out_3_fm[row][col][out_dep] = 0
                #print('3rd layer output: ',layer_out_2_fm[row][col][out_dep])
                #print('activation output: ',layer_out_3_fm[row][col][out_dep])
            else:
                layer_out_3_fm[row][col][out_dep] = layer_out_2_fm[row][col][out_dep]
                #print('3rd layer output: ',layer_out_2_fm[row][col][out_dep])
                #print('activation output: ',layer_out_3_fm[row][col][out_dep])

#5th layer, max pooling
pool_stride = 2
layer_out_4_fm = np.zeros((12,12,32),dtype=np.float32) #initialize output
for out_dep in range(32): #output put depth, number of filters
    for row in range(12): #output row
        for col in range(12): #output column
            temp = [layer_out_3_fm[pool_stride*row+0][pool_stride*col+0][out_dep],
                    layer_out_3_fm[pool_stride*row+0][pool_stride*col+1][out_dep],
                    layer_out_3_fm[pool_stride*row+1][pool_stride*col+0][out_dep],
                    layer_out_3_fm[pool_stride*row+1][pool_stride*col+1][out_dep]]
            layer_out_4_fm[row][col][out_dep] = max(temp)

#8th layer, dense 1
layer_out_7_fm = np.zeros(128,dtype=np.float32) #initialize output
for i in range(128): #initialize the output with biases
    layer_out_7_fm[i] = biases_7[i]
    for row in range(12): #output row
        for col in range(12): #output column
            for out_dep in range(32): #output put depth, number of filters
                layer_out_7_fm[i] += layer_out_4_fm[row][col][out_dep]*weights_7[((row*12+col)*32+out_dep)*128+i]
                #In a 3D matrix the value at index (k,h,w) is pysically located at index (k * H + h) * W + w.

#9th layer, activation 3, relu
layer_out_8_fm = np.zeros(128,dtype=np.float32) #initialize output
for i in range(128):
    if layer_out_7_fm[i] < 0:
        layer_out_8_fm[i] = 0;
    else:
        layer_out_8_fm[i] = layer_out_7_fm[i]

#11th layer, dense 2
layer_out_10_fm = np.zeros(10,dtype=np.float32) #initialize output
for i in range(10): #initialize the output with biases
    layer_out_10_fm[i] = biases_10[i]
    for j in range(128): #output row
        layer_out_10_fm[i] += layer_out_8_fm[j]*weights_10[j*10+i]
        
#12th layer, activation 4
layer_out_11_fm = softmax(layer_out_10_fm)

print('the final output',layer_out_11_fm)
print('golden output',layer_out_11)

#----------------------------------------------------------------------------
#           below is the code for debug purpose only
#----------------------------------------------------------------------------
#test if we get the correct output
#print('out:',"%.10f" % layer_out_0_fm[row][col][out_dep],'golden:%.10f', "%.10f" %layer_out_0[row][col][out_dep])
#print('out type:',layer_out_0_fm.dtype,'golden type:',layer_out_0.dtype)
#print('weights type:',weights.dtype,'biases type:',biases.dtype)
#print('input type:', im_in.dtype)

#test first layer's output(conv)
#for out_dep in range(32): #output put depth, number of filters
#    for row in range(26): #output row
#        for col in range(26): #output column
#            if(layer_out_0_fm[row][col][out_dep]!=layer_out_0[row][col][out_dep]):
#               print('mismatch at position', row, col, out_dep)
#               print('out:','%.10f' % layer_out_0_fm[row][col][out_dep],'golden: ','%.10f' % layer_out_0[row][col][out_dep])

#test second layer's output(activation)
#for out_dep in range(32): #output put depth, number of filters
#    for row in range(26): #output row
#        for col in range(26): #output column#            if(layer_out_1_fm[row][col][out_dep]!=layer_out_1[row][col][out_dep]):
#            if layer_out_1_fm[row][col][out_dep] < 0:
#                print('out:',layer_out_1_fm[row][col][out_dep],'golden: ',layer_out_1[row][col][out_dep])

#test third layer's output(conv)
#for out_dep in range(32): #output put depth, number of filters
#    for row in range(24): #output row
#        for col in range(24): #output column
#            if(layer_out_2_fm[row][col][out_dep]!=layer_out_2[row][col][out_dep]):
#                print('mismatch at position', row, col, out_dep)
#                print('out:',layer_out_2_fm[row][col][out_dep],'golden: ',layer_out_2[row][col][out_dep])

#test 4th layer's output(activation)
#for out_dep in range(32): #output put depth, number of filters
#    for row in range(24): #output row
#        for col in range(24): #output column#            if(layer_out_1_fm[row][col][out_dep]!=layer_out_1[row][col][out_dep]):
#            if layer_out_1_fm[row][col][out_dep] < 0:
#                print('out:',layer_out_3_fm[row][col][out_dep],'golden: ',layer_out_3[row][col][out_dep])

#test 5th layer's output(max pooling)
#for out_dep in range(32): #output put depth, number of filters
#    for row in range(12): #output row
#        for col in range(12): #output column
#            if layer_out_4_fm[row][col][out_dep]!=layer_out_4[row][col][out_dep]:
#                print('mismatch at position',row,col,out_dep)
#                print('out:',layer_out_4_fm[row][col][out_dep],'golden:',layer_out_4[row][col][out_dep])

#test 8th layer's output(dense 1)
#for i in range(128):
#    if layer_out_7_fm[i] != layer_out_7[i]:
#        print('mistmatch at position ',i)
#        print('out: ',layer_out_7_fm[i],'golden: ',layer_out_7[i])

#test 11th layer's output(dense 2)
#for i in range(10):
#    if layer_out_10_fm[i] != layer_out_10[i]:
#        print('mistmatch at position ',i)
#        print('out: ',layer_out_10_fm[i],'golden: ',layer_out_10[i])
                
#if(np.array_equal(layer_out_0_fm,layer_out_0)):
#    print('succeed!')
#else:
#    print('fail!')


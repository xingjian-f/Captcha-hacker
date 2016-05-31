from keras.models import Graph
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from util import weight_crossentropy


def build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes):
    model = Graph()
    model.add_input(name='input', input_shape=(img_channels, img_width, img_height))
    # 1 conv
    model.add_node(Convolution2D(48, 5, 5, border_mode='same', activation='relu', 
        # W_regularizer='l2'
        ), input='input', name='conv1')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2,2)), input='conv1', name='pool1')
    model.add_node(Dropout(0.5), input='pool1', name='drop1')
    model.add_node(BatchNormalization(axis=1), input='drop1', name='norm1')
    # 2 conv0
    model.add_node(Convolution2D(64, 5, 5, border_mode='same', activation='relu',
        # W_regularizer='l2', 
        ), input='norm1', name='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv2', name='pool2')
    model.add_node(Dropout(0.5), input='pool2', name='drop2')
    model.add_node(BatchNormalization(axis=1), input='drop2', name='norm2')
    # 3 conv
    model.add_node(Convolution2D(128, 5, 5, border_mode='same', activation='relu',
        # W_regularizer='l2', 
        ), input='norm2', name='conv3')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2,2)), input='conv3', name='pool3')
    model.add_node(Dropout(0.5), input='pool3', name='drop3')
    model.add_node(BatchNormalization(axis=1), input='drop3', name='norm3')
    # 4 conv
    model.add_node(Convolution2D(160, 5, 5, border_mode='same', activation='relu',
        # W_regularizer='l2', 
        ), input='norm3', name='conv4')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv4', name='pool4')
    model.add_node(Dropout(0.5), input='pool4', name='drop4')
    model.add_node(BatchNormalization(axis=1), input='drop4', name='norm4')
    # 5 conv
    model.add_node(Convolution2D(192, 5, 5, border_mode='same', activation='relu',
        # W_regularizer='l2', 
        ), input='norm4', name='conv5')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv5', name='pool5')
    model.add_node(Dropout(0.5), input='pool5', name='drop5')
    model.add_node(BatchNormalization(axis=1), input='drop5', name='norm5')
    # 6 conv
    model.add_node(Convolution2D(192, 5, 5, border_mode='same', activation='relu',
        # W_regularizer='l2', 
        ), input='norm5', name='conv6')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv6', name='pool6')
    model.add_node(Dropout(0.5), input='pool6', name='drop6')
    model.add_node(BatchNormalization(axis=1), input='drop6', name='norm6')
    # 7 conv
    model.add_node(Convolution2D(192, 5, 5, border_mode='same', activation='relu',
        # W_regularizer='l2', 
        ), input='norm6', name='conv7')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv7', name='pool7')
    model.add_node(Dropout(0.5), input='pool7', name='drop7')
    model.add_node(BatchNormalization(axis=1), input='drop7', name='norm7')
    # 8 conv
    model.add_node(Convolution2D(192, 5, 5, border_mode='same', activation='relu',
        # W_regularizer='l2', 
        ), input='norm7', name='conv8')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv8', name='pool8')
    model.add_node(Dropout(0.5), input='pool8', name='drop8')
    model.add_node(BatchNormalization(axis=1), input='drop8', name='norm8')
    # 1 Dense 
    model.add_node(Flatten(), input='norm8', name='flat')
    model.add_node(Dense(512, activation='relu', 
        W_regularizer='l2',
        ), input='flat', name='dense')
    model.add_node(Dropout(0.5), input='dense', name='drop9')
    model.add_node(BatchNormalization(axis=1), input='drop9', name='norm9')
    # 2 Dense
    for i in range(1, nb_classes+1):
        model.add_node(Dense(1, activation='sigmoid'), input='norm9', name='dense%d' % i)
    for i in range(1, nb_classes+1):
        model.add_output(name='output%d' % i, input='dense%d' % i)

    loss = {'output%d'%i:weight_crossentropy for i in range(1, nb_classes+1)}
    metrics=['accuracy']
    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    return model
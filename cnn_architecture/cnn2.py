from keras.models import Graph
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from util import log_prob

def build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes):
    model = Graph()
    model.add_input(name='input', input_shape=(img_channels, img_width, img_height))
    # 1 conv
    model.add_node(Convolution2D(48, 5, 5, border_mode='same', activation='relu',
        W_regularizer='l2'
        ), input='input', name='conv1')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2,2)), input='conv1', name='pool1')
    model.add_node(Dropout(0.5), input='pool1', name='drop1')
    # model.add_node(GaussianNoise(0.15), input='drop1', name='noise1')
    b1 = BatchNormalization(axis=1, mode=0)
    model.add_node(b1, input='drop1', name='norm1')
    # 2 conv
    model.add_node(Convolution2D(64, 5, 5, border_mode='same', activation='relu',
        W_regularizer='l2'
        ), input='norm1', name='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv2', name='pool2')
    model.add_node(Dropout(0.5), input='pool2', name='drop2')
    model.add_node(BatchNormalization(axis=1,mode=0), input='drop2', name='norm2')
    # 3 conv
    model.add_node(Convolution2D(128, 5, 5, border_mode='same', activation='relu',
        W_regularizer='l2'
        ), input='norm2', name='conv3')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2,2)), input='conv3', name='pool3')
    model.add_node(Dropout(0.5), input='pool3', name='drop3')
    model.add_node(BatchNormalization(axis=1,mode=0), input='drop3', name='norm3')
    # 4 conv
    model.add_node(Convolution2D(160, 5, 5, border_mode='same', activation='relu',
        W_regularizer='l2'
        ), input='norm3', name='conv4')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv4', name='pool4')
    model.add_node(Dropout(0.5), input='pool4', name='drop4')
    model.add_node(BatchNormalization(axis=1, mode=0), input='drop4', name='norm4')
    # 1 Dense
    model.add_node(Flatten(), input='norm4', name='flat')
    model.add_node(Dense(512, activation='relu',
        W_regularizer='l2'
        ), input='flat', name='dense')
    model.add_node(Dropout(0.5), input='dense', name='drop5')
    model.add_node(BatchNormalization(axis=1,mode=0), input='drop5', name='norm5')
    # 2 Dense
    model.add_node(Dense(max_nb_cha, activation='softmax'), input='norm5', name='dense_nb')
    for i in range(1, max_nb_cha+1):
        model.add_node(Dense(nb_classes, activation='softmax'), input='norm5', name='dense%d' % i)
    model.add_output(name='output_nb', input='dense_nb')
    for i in range(1, max_nb_cha+1):
        model.add_output(name='output%d' % i, input='dense%d' % i)

    loss = {'output%d'%i:'categorical_crossentropy' for i in range(1, max_nb_cha+1)}
    loss['output_nb'] = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    return model

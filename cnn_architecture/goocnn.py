from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes):
    model = Graph()
    model.add_input(name='input', input_shape=(img_channels, img_width, img_height))
    # 1 conv
    model.add_node(Convolution2D(48, 5, 5, border_mode='same', activation='relu'), input='input', name='conv1')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2,2)), input='conv1', name='pool1')
    model.add_node(Dropout(0.25), input='pool1', name='drop1')
    # 2 conv
    model.add_node(Convolution2D(64, 5, 5, border_mode='same', activation='relu'), input='drop1', name='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv2', name='pool2')
    model.add_node(Dropout(0.25), input='pool2', name='drop2')
    # 3 conv
    model.add_node(Convolution2D(128, 5, 5, border_mode='same', activation='relu'), input='drop2', name='conv3')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2,2)), input='conv3', name='pool3')
    model.add_node(Dropout(0.25), input='pool3', name='drop3')
    # 4 conv
    model.add_node(Convolution2D(160, 5, 5, border_mode='same', activation='relu'), input='drop3', name='conv4')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv4', name='pool4')
    model.add_node(Dropout(0.25), input='pool4', name='drop4')
    # 5 conv
    model.add_node(Convolution2D(192, 5, 5, border_mode='same', activation='relu'), input='drop4', name='conv5')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2,2)), input='conv5', name='pool5')
    model.add_node(Dropout(0.25), input='pool5', name='drop5')
    # 6 conv
    model.add_node(Convolution2D(192, 5, 5, border_mode='same', activation='relu'), input='drop5', name='conv6')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv6', name='pool6')
    model.add_node(Dropout(0.25), input='pool6', name='drop6')
    # 7 conv
    model.add_node(Convolution2D(192, 5, 5, border_mode='same', activation='relu'), input='drop6', name='conv7')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2,2)), input='conv7', name='pool7')
    model.add_node(Dropout(0.25), input='pool7', name='drop7')
    # 8 conv
    model.add_node(Convolution2D(192, 5, 5, border_mode='same', activation='relu'), input='drop7', name='conv8')
    model.add_node(MaxPooling2D(pool_size=(2, 2), strides=(1,1)), input='conv8', name='pool8')
    model.add_node(Dropout(0.25), input='pool8', name='drop8')
    # 1 Dense 
    model.add_node(Flatten(), input='drop8', name='flat')
    model.add_node(Dense(3072, activation='relu'), input='flat', name='dense')
    model.add_node(Dropout(0.5), input='dense', name='drop9')
    # 2 Dense
    model.add_node(Dense(max_nb_cha, activation='softmax'), input='drop9', name='dense_nb')
    for i in range(1, max_nb_cha+1):
        model.add_node(Dense(nb_classes, activation='softmax'), input='drop9', name='dense%d' % i)
    model.add_output(name='output_nb', input='dense_nb')
    for i in range(1, max_nb_cha+1):
        model.add_output(name='output%d' % i, input='dense%d' % i)

    loss = {'output%d'%i:'categorical_crossentropy' for i in range(1, max_nb_cha+1)}
    loss['output_nb'] = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer='adadelta')
    return model
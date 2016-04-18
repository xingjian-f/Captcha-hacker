from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes):
    model = Graph()
    model.add_input(name='input', input_shape=(img_channels, img_width, img_height))
    model.add_node(Convolution2D(32, 5, 5, border_mode='same', activation='relu'), input='input', name='conv1')
    model.add_node(Convolution2D(32, 5, 5, activation='relu'), input='conv1', name='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2)), input='conv2', name='pool1')
    model.add_node(Dropout(0.25), input='pool1', name='drop1')

    model.add_node(Convolution2D(64, 5, 5, border_mode='same', activation='relu'), input='drop1', name='conv3')
    model.add_node(Convolution2D(64, 5, 5, activation='relu'), input='conv3', name='conv4')
    model.add_node(MaxPooling2D(pool_size=(2, 2)), input='conv4', name='pool2')
    model.add_node(Dropout(0.25), input='pool2', name='drop2')

    model.add_node(Flatten(), input='drop2', name='flat')
    model.add_node(Dense(512, activation='relu'), input='flat', name='dense')
    model.add_node(Dropout(0.5), input='dense', name='drop3')

    model.add_node(Dense(max_nb_cha, activation='softmax'), input='drop3', name='dense_nb')
    for i in range(1, max_nb_cha+1):
        model.add_node(Dense(nb_classes, activation='softmax'), input='drop3', name='dense%d' % i)
    model.add_output(name='output_nb', input='dense_nb')
    for i in range(1, max_nb_cha+1):
        model.add_output(name='output%d' % i, input='dense%d' % i)

    loss = {'output%d'%i:'categorical_crossentropy' for i in range(1, max_nb_cha+1)}
    loss['output_nb'] = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer='adadelta')
    return model
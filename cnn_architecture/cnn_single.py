from keras.models import Graph, Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from util import log_prob


def build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes):
    model = Sequential()
    # 1 conv
    model.add(Convolution2D(48, 5, 5, border_mode='same', activation='relu', 
        W_regularizer='l2',
        input_shape=(img_channels, img_width, img_height)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(axis=1))
    # 2 conv
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu',
        W_regularizer='l2',
        ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(axis=1))
    # 3 conv
    model.add(Convolution2D(128, 5, 5, border_mode='same', activation='relu',
        W_regularizer='l2',
        ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(axis=1))
    # 4 conv
    model.add(Convolution2D(160, 5, 5, border_mode='same', activation='relu',
        W_regularizer='l2', 
        ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(axis=1))
    # 1 Dense 
    model.add(Flatten())
    model.add(Dense(512, activation='relu', 
        W_regularizer='l2', 
        ))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(axis=1))
    # 2 Dense
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
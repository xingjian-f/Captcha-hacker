#coding:utf-8
import os
import time
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from util import one_hot_decoder
from load_data import load_data


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
    model.add_node(Dense(nb_classes, activation='softmax'), input='drop3', name='dense1')
    model.add_node(Dense(nb_classes, activation='softmax'), input='drop3', name='dense2')
    model.add_node(Dense(nb_classes, activation='softmax'), input='drop3', name='dense3')
    model.add_node(Dense(nb_classes, activation='softmax'), input='drop3', name='dense4')
    model.add_output(name='output_nb', input='dense_nb')
    model.add_output(name='output1', input='dense1')
    model.add_output(name='output2', input='dense2')
    model.add_output(name='output3', input='dense3')
    model.add_output(name='output4', input='dense4')

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss={'output_nb':'categorical_crossentropy', 'output1':'categorical_crossentropy',
                'output2':'categorical_crossentropy', 'output3':'categorical_crossentropy',
                'output4':'categorical_crossentropy'}, 
                optimizer='adadelta')
    return model


def test(model, len_set, cha_set, max_nb_cha, X_test, Y_test_nb, Y_test):
    # 开始预测并验证准确率，需要先把预测结果从概率转到对应的标签
    predictions = model.predict({'input':X_test})
    pred_nbs = one_hot_decoder(predictions['output_nb'], len_set)
    pred_chas = [one_hot_decoder(predictions['output%d' % j], cha_set) for j in range(1, 5)]
    Y_test_nb = one_hot_decoder(Y_test_nb, len_set)
    Y_test = [one_hot_decoder(i, cha_set) for i in Y_test]

    correct = 0
    len_correct = 0
    nb_sample = X_test.shape[0]
    for i in range(nb_sample):
        pred_nb = pred_nbs[i]
        true_nb = Y_test_nb[i]
        # print 'len:', pred_nb, true_nb
        allright = (pred_nb == true_nb)
        if allright:
            len_correct += 1
        for j in range(true_nb):
            # print pred_chas[j][i], Y_test[j][i]
            allright = allright and (pred_chas[j][i] == Y_test[j][i])
        if allright:
            correct += 1
    print 'Length accuracy:', float(len_correct) / nb_sample
    print 'Accuracy:', float(correct) / nb_sample


def train(model, batch_size, nb_classes, max_nb_cha, nb_epoch, save_dir, save_minutes, X_train, Y_train_nb, Y_train):
    print 'X_train shape:', X_train.shape
    print X_train.shape[0], 'train samples'

    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    data = {'input':X_train, 'output_nb':Y_train_nb, 'output1':Y_train[0], 'output2':Y_train[1],
        'output3':Y_train[2], 'output4':Y_train[3]}
    tag = time.time()
    start_time = time.time()
    for i in range(nb_epoch):
        model.fit(data, batch_size=batch_size, nb_epoch=1, validation_split=0.3)
        if time.time()-tag > save_minutes*60:
            save_path = save_dir + str(datetime.now()).split('.')[0].split()[1] + '.h5' # 存储路径使用当前的时间
            model.save_weights(save_path)
            tag = time.time() # 重新为存储计时

    save_path = save_dir + str(datetime.now()).split('.')[0].split()[1] + '.h5'
    model.save_weights(save_path)
    print 'Training time(h):', (time.time()-start_time) / 3600

if __name__ == '__main__':
    img_width, img_height = 200, 50
    img_channels = 3
    max_nb_cha = 6 # 文本最大长度
    len_set = range(1, max_nb_cha+1) # 文本可能的长度
    cha_set = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['empty'] # 文本字符集
    nb_classes = 63 # 数字10 + 大小写字母52 + empty1
    batch_size = 32
    nb_epoch = 20
    save_minutes = 5 # 每隔多少分钟保存一次模型

    save_dir = 'model/' + str(datetime.now()).split('.')[0].split()[0] + '/' # 每天都保存在相应的目录中
    train_data_dir = 'gen_images/img_data'
    # train_data_dir = 'gen_images/img_data/00000000'
    test_data_dir = 'test_data/'
    weights_file_path = 'model/2016-04-15/19:49:18.h5'

    model = build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes) # 生成CNN的架构
    
    model.load_weights(weights_file_path) # 读取训练好的模型
    X_train, Y_train_nb, Y_train = load_data(train_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set) 
    train(model, batch_size, nb_classes, max_nb_cha, nb_epoch, save_dir, save_minutes, X_train, Y_train_nb, Y_train)
    
    # X_test, Y_test_nb, Y_test = load_data(test_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set)
    X_test, Y_test_nb, Y_test = X_train, Y_train_nb, Y_train
    test(model, len_set, cha_set, max_nb_cha, X_test, Y_test_nb, Y_test)
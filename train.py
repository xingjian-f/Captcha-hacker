#coding:utf-8
import os
import time
from datetime import datetime
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from util import one_hot_decoder, plot_loss_figure, pack_data
from load_data import load_data, generate_data
# from cnn_architecture.cnn0 import build_cnn
from cnn_architecture.cnn1 import build_cnn

def test(model, len_set, cha_set, max_nb_cha, X_test, Y_test_nb, Y_test):
    # 开始预测并验证准确率，需要先把预测结果从概率转到对应的标签
    predictions = model.predict({'input':X_test})
    pred_nbs = one_hot_decoder(predictions['output_nb'], len_set)
    pred_chas = [one_hot_decoder(predictions['output%d' % j], cha_set) for j in range(1, max_nb_cha+1)]
    Y_test_nb = one_hot_decoder(Y_test_nb, len_set)
    Y_test = [one_hot_decoder(i, cha_set) for i in Y_test]

    correct = 0
    len_correct = 0
    nb_sample = X_test.shape[0]
    for i in range(nb_sample):
        pred_nb = pred_nbs[i]
        true_nb = Y_test_nb[i]
        print 'len(pred, true):', pred_nb, true_nb
        allright = (pred_nb == true_nb)
        if allright:
            len_correct += 1
        for j in range(true_nb):
            print pred_chas[j][i], Y_test[j][i]
            allright = allright and (pred_chas[j][i] == Y_test[j][i])
        if allright:
            correct += 1
    print 'Length accuracy:', float(len_correct) / nb_sample
    print 'Accuracy:', float(correct) / nb_sample


def train(model, batch_size, max_nb_cha, nb_epoch, save_dir, train_data, val_data):
    print 'X_train shape:', X_train.shape
    print X_train.shape[0], 'train samples'
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)

    start_time = time.time()
    save_path = save_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    check_pointer = ModelCheckpoint(save_path)
    history = model.fit(train_data, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=val_data, 
    	callbacks=[check_pointer])

    plot_loss_figure(history, save_dir + str(datetime.now()).split('.')[0].split()[1]+'.jpg')
    print 'Training time(h):', (time.time()-start_time) / 3600


def train_on_generator(model, batch_size, max_nb_cha, nb_epoch, save_dir, generator, val_data):
	print 'Train using generator'
	if os.path.exists(save_dir) == False:
		os.mkdir(save_dir)
	
	start_time = time.time()
	save_path = save_dir + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
	check_pointer = ModelCheckpoint(save_path)
	samples_per_epoch = 50000 # 每个epoch跑多少数据
	history = model.fit_generator(generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
		nb_worker=4, validation_data=val_data, callbacks=[check_pointer])

	plot_loss_figure(history, save_dir + str(datetime.now()).split('.')[0].split()[1]+'.jpg')
	print 'Training time(h):', (time.time()-start_time) / 3600

    
if __name__ == '__main__':
    img_width, img_height = 200, 50
    img_channels = 3
    max_nb_cha = 6 # 文本最大长度
    len_set = range(1, max_nb_cha+1) # 文本可能的长度
    cha_set = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['empty'] # 文本字符集
    nb_classes = 63 # 数字10 + 大小写字母52 + empty1
    batch_size = 128
    nb_epoch = 40

    save_dir = 'model/' + str(datetime.now()).split('.')[0].split()[0] + '/' # 模型保存在当天对应的目录中
    train_data_dir = 'gen_images/img_data/train'
    # train_data_dir = 'gen_images/img_data/00000010'
    val_data_dir = 'gen_images/img_data/validation'
    # val_data_dir = 'gen_images/img_data/00000010'
    test_data_dir = 'test_data/'
    weights_file_path = 'model/2016-04-20/weights.39-2.22.hdf5'

    model = build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes) # 生成CNN的架构
    model.load_weights(weights_file_path) # 读取训练好的模型

    # 先读取整个数据集，然后训练    
    # X_val, Y_val_nb, Y_val = load_data(val_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set)
    # val_data = pack_data(X_val, Y_val_nb, Y_val, max_nb_cha)
    # X_train, Y_train_nb, Y_train = load_data(train_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set) 
    # train_data = pack_data(X_train, Y_train_nb, Y_train, max_nb_cha)
    # train(model, batch_size, max_nb_cha, nb_epoch, save_dir, train_data, val_data)
    # 内存一次性放不下，则边训练边读取数据, 问题是太慢！
    # generator = generate_data(train_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set, batch_size)
    # X_val, Y_val_nb, Y_val = load_data(val_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set)
    # val_data = pack_data(X_val, Y_val_nb, Y_val, max_nb_cha)
    # train_on_generator(model, batch_size, max_nb_cha, nb_epoch, save_dir, generator, val_data)

    X_test, Y_test_nb, Y_test = load_data(test_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set)
    # X_test, Y_test_nb, Y_test = X_val, Y_val_nb, Y_val
    test(model, len_set, cha_set, max_nb_cha, X_test, Y_test_nb, Y_test)
    # X_test, Y_test_nb, Y_test = X_train, Y_train_nb, Y_train
    # test(model, len_set, cha_set, max_nb_cha, X_test, Y_test_nb, Y_test)
#coding:utf-8
import os
import time
from datetime import datetime
from keras.optimizers import SGD
from util import one_hot_decoder, plot_loss_figure
from load_data import load_data, generate_data
from cnn_architecture.cnn0 import build_cnn
# from cnn_architecture.goocnn import build_cnn

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


def train(model, batch_size, max_nb_cha, nb_epoch, save_dir, save_minutes, X_train, Y_train_nb, Y_train):
    print 'X_train shape:', X_train.shape
    print X_train.shape[0], 'train samples'

    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    data = {'output%d'%i:Y_train[i-1] for i in range(1, max_nb_cha+1)}
    data['input'] = X_train
    data['output_nb'] = Y_train_nb
    tag = time.time()
    start_time = time.time()
    history = []
    for i in range(nb_epoch):
        history.append(model.fit(data, batch_size=batch_size, nb_epoch=1, validation_split=0.3))
        if time.time()-tag > save_minutes*60:
            save_path = save_dir + str(datetime.now()).split('.')[0].split()[1] + '.h5' # 存储路径使用当前的时间
            model.save_weights(save_path)
            tag = time.time() # 重新为存储计时

    save_path = save_dir + str(datetime.now()).split('.')[0].split()[1] + '.h5'
    model.save_weights(save_path)
    plot_loss_figure(history, save_dir + str(datetime.now()).split('.')[0].split()[1]+'.jpg')
    print 'Training time(h):', (time.time()-start_time) / 3600


def train_on_generator(model, batch_size, max_nb_cha, nb_epoch, save_dir, save_minutes, generator):
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    tag = time.time()
    start_time = time.time()
    history = []
    for i in range(nb_epoch):
        samples_per_epoch = 100 # 每个epoch跑多少数据
        history.append(model.fit_generator(generator, samples_per_epoch=samples_per_epoch, nb_epoch=1, 
            nb_worker=4, validation_data=generator, nb_val_samples=50))
        if time.time()-tag > save_minutes*60:
            save_path = save_dir + str(datetime.now()).split('.')[0].split()[1] + '.h5' # 存储路径使用当前的时间
            model.save_weights(save_path)
            tag = time.time() # 重新为存储计时

    save_path = save_dir + str(datetime.now()).split('.')[0].split()[1] + '.h5'
    model.save_weights(save_path)
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
    nb_epoch = 10
    save_minutes = 5 # 每隔多少分钟保存一次模型

    save_dir = 'model/' + str(datetime.now()).split('.')[0].split()[0] + '/' # 模型保存在当天对应的目录中
    # train_data_dir = 'gen_images/img_data/all'
    train_data_dir = 'gen_images/img_data/00000000'
    test_data_dir = 'test_data/'
    weights_file_path = 'model/2016-04-19/14:26:25.h5'

    model = build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes) # 生成CNN的架构
    model.load_weights(weights_file_path) # 读取训练好的模型

    # 先生成整个数据集，然后训练
    X_train, Y_train_nb, Y_train = load_data(train_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set) 
    train(model, batch_size, max_nb_cha, nb_epoch, save_dir, save_minutes, X_train, Y_train_nb, Y_train)
    # 边训练边生成数据
    # generator = generate_data(train_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set, batch_size)
    # train_on_generator(model, batch_size, max_nb_cha, nb_epoch, save_dir, save_minutes, generator)

    X_test, Y_test_nb, Y_test = load_data(test_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set)
    # X_test, Y_test_nb, Y_test = X_train, Y_train_nb, Y_train
    test(model, len_set, cha_set, max_nb_cha, X_test, Y_test_nb, Y_test)
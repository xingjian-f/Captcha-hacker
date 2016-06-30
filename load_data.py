#coding:utf-8
import time
import os
import numpy as np
from PIL import Image
from util import one_hot_encoder


def load_data(input_dir, max_nb_cha, width, height, channels, len_set, cha_set):
    """
    数据文件夹需严格符合，图片文件(命名为number.jpg)及一个label.txt 这种形式
    # y[0],[1],[2],[3] 分别对应第1,2,3,4个字符类推
    """
    print 'Loading data...'
    tag = time.time()
    x = []
    y_nb = []
    y = [[] for i in range(max_nb_cha)]

    for dirpath, dirnames, filenames in os.walk(input_dir):
        nb_pic = len(filenames)-1
        if nb_pic >= 1:
            cnt = 50000 ## pay attention!!!!
            for i in range(1, nb_pic+1):
                cnt -= 1
                if cnt == 0:
                    break
                filename = str(i) + '.jpg'
                filepath = dirpath + os.sep + filename
                im = Image.open(filepath)
                im = im.resize((width, height))
                pixels = list(im.getdata())
                if channels > 1:
                    x.append([[[pixels[k*width+i][j] for k in range(height)] for i in range(width)] for j in range(channels)]) # 转成（channel，width，height）shape
                else:
                    x.append([[[pixels[k*width+i] for k in range(height)] for i in range(width)]])
            
            label_path = dirpath + os.sep + 'label.txt'
            with open(label_path) as f:
                cnt = 50000
                for raw in f:
                    # print raw
                    cnt -= 1
                    if cnt == 0:
                        break
                    raw = raw.strip('\n\r')
                    if len(raw) > 0:
                        y_nb.append(len(raw))
                        for i in range(max_nb_cha):
                            if i < len(raw):
                                y[i].append(raw[i])
                            else:
                                y[i].append('empty')
                    

    # 转成keras能接受的数据形式，以及做one hot 编码
    x = np.array(x)
    x = x.astype('float32') # gpu只接受32位浮点运算
    x /= 255 # normalized
    y_nb = np.array(one_hot_encoder(y_nb, len_set))
    for i in range(max_nb_cha):
        y[i] = np.array(one_hot_encoder(y[i], cha_set))

    print 'Data loaded, spend time(m) :', (time.time()-tag)/60
    return [x, y_nb, y]


def generate_data(input_dir, max_nb_cha, width, height, channels, len_set, cha_set, batch_size):
    cnt = 0
    x = []
    y_nb = []
    y = [[] for i in range(max_nb_cha)]
    while True:
        for dirpath, dirnames, filenames in os.walk(input_dir):
            nb_pic = len(filenames)-1
            if nb_pic >= 1:            
                label_path = dirpath + os.sep + 'label.txt'
                with open(label_path) as f:
                    for (i, raw) in zip(range(1, nb_pic+1), f):
                        filename = str(i) + '.jpg'
                        filepath = dirpath + os.sep + filename
                        im = Image.open(filepath)
                        im = im.resize((width, height))
                        pixels = list(im.getdata())
                        x.append([[[pixels[k*width+i][j] for k in range(height)] for i in range(width)] for j in range(channels)]) # 转成（channel，width，height）shape
                        
                        raw = raw.strip('\n\r')
                        y_nb.append(len(raw))
                        for i in range(max_nb_cha):
                            if i < len(raw):
                                y[i].append(raw[i])
                            else:
                                y[i].append('empty')

                        cnt += 1
                        if cnt == batch_size:
                            x = np.array(x)
                            x = x.astype('float32') # gpu只接受32位浮点运算
                            x /= 255 # normalized
                            y_nb = np.array(one_hot_encoder(y_nb, len_set))
                            for i in range(max_nb_cha):
                                y[i] = np.array(one_hot_encoder(y[i], cha_set))

                            ret = {'output%d'%i:y[i-1] for i in range(1, max_nb_cha+1)}
                            ret['input'] = x
                            ret['output_nb'] = y_nb
                            yield ret
                            x = []
                            y_nb = []
                            y = [[] for i in range(max_nb_cha)]
                            cnt = 0


if __name__ == '__main__':
    input_dir = 'gen_images/img_data/'
    max_nb_cha = 6
    width = 200
    height = 50
    channels = 3
    len_set = range(1, max_nb_cha+1)
    cha_set = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['empty'] # 文本字符集
    # x = load_data(input_dir, max_nb_cha, width, height, channels, len_set, cha_set)
    turn = 2
    batch_size = 4
    for x in generate_data(input_dir, max_nb_cha, width, height, channels, len_set, cha_set, batch_size):
        print x
        turn -= 1
        if turn <= 0:
            break
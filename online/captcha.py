# coding:utf-8
import sys
import os
import time
sys.path.append('../')
import json
import numpy as np 
from PIL import Image 
from cnn_architecture.cnn2 import build_cnn
from util import one_hot_decoder

def init_model():
    img_width, img_height = 223, 50
    img_channels = 3 
    max_nb_cha = 6 # 文本最大长度
    nb_classes = 36
    weights_file_path = '/home/wiking/Captcha-hacker/model/2016-06-22/weights.24-2.52.hdf5'
    model = build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes) # 生成CNN的架构
    model.load_weights(weights_file_path) # 读取训练好的模型
    return model

def load_data(imgdata, width, height, channels):
    img = Image.open(imgdata)
    im = img.resize((width, height))
    pixels = list(im.getdata())
    x = []
    if channels > 1:
        x.append([[[pixels[k*width+i][j] for k in range(height)] for i in range(width)] for j in range(channels)]) # 转成（channel，width，height）shape
    else:
        x.append([[[pixels[k*width+i] for k in range(height)] for i in range(width)]])
    x = np.array(x)
    x = x.astype('float32') # gpu只接受32位浮点运算
    x /= 255 # normalized
    return x

def predict(model, imgdata):
    img_width, img_height = 223, 50
    img_channels = 3 
    max_nb_cha = 6 # 文本最大长度
    len_set = range(1, max_nb_cha+1) # 文本可能的长度
    cha_set = list("0123456789qwertyuioplkjhgfdsazxcvbnm")# 文本字符集
    nb_classes = 36

    X_test = load_data(imgdata, img_width, img_height, img_channels)
    # print X_test
    predictions = model.predict({'input':X_test})
    pred_chas = [one_hot_decoder(predictions['output%d' % j], cha_set) for j in range(1, max_nb_cha+1)]
    pred_nbs = one_hot_decoder(predictions['output_nb'], len_set)
    length = min(max_nb_cha, pred_nbs[0])
    ans = []
    for i in range(length):
        ans.append(pred_chas[i][0])
    ans = ''.join(ans)
    # print ans
    # print predictions['output1']
    res = {'valid':True, 'answer':ans, 'expr':ans}
    
    return json.dumps(res)
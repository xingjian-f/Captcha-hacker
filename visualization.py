# coding: utf-8
import os
import random
import theano
from keras import backend as K
from PIL import Image
import numpy as np
from cnn_architecture.cnn2 import build_cnn
from load_data import load_data
from train import test

def visualize(model, datas):
	for img_name in range(len(datas)):
		data = [datas[img_name]]
		layers = ['input', 'conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4', 'pool4']
		save_dir = 'visual_pic/%d/'%(img_name+1)
		if os.path.exists(save_dir) == False:
			os.mkdir(save_dir)
		for layer_name in layers:
			layer = K.function([model.input, K.learning_phase()], [model.get_layer(layer_name).output])
			layer_out = layer([data, 0])[0][0]
			nb_filter = layer_out.shape[0]
			layer_dir = save_dir +layer_name
			for k in range(nb_filter):
				img_data = layer_out[k]
				width, height = img_data.shape[0], img_data.shape[1]
				max_val = 1e-9
				for i in range(width):
					for j in range(height):
						max_val = max(max_val, img_data[i][j])
				im = Image.new('L', img_data.shape)
				for i in range(width):
					for j in range(height):
						val = int(img_data[i][j] / max_val * 255)# scale pixel value back to 255
						im.putpixel((i,j), val)
				if os.path.exists(layer_dir) == False:
					os.mkdir(layer_dir)
				save_path = layer_dir+'/%d.jpg'%k
				im.save(save_path)
			print layer_dir


if __name__ == '__main__':
	img_width, img_height = 200, 50
	img_channels = 3 
	max_nb_cha = 6 # 文本最大长度
	len_set = range(1, max_nb_cha+1) # 文本可能的长度
	cha_set = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['empty'] # 文本字符集
	nb_classes = 63 # 数字10 + 大小写字母52 + empty1
	
	visual_data_dir = 'test_data/jiangsu'
	# visual_data_dir = 'gen_images/img_data/visual'
	weights_file_path = 'model/2016-04-23/weights.17-1.58.hdf5'
	
	model = build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes) # 生成CNN的架构
	model.load_weights(weights_file_path) # 读取训练好的模型
	
	X, Y_nb, Y = load_data(visual_data_dir, max_nb_cha, img_width, img_height, img_channels, len_set, cha_set)
	# X = np.array([random.choice(X)])
	visualize(model, X)
	# test(model, len_set, cha_set, max_nb_cha, X, Y_nb, Y)
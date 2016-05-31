import os
import time
from datetime import datetime
from util import one_hot_decoder, plot_loss_figure, pack_data
from cnn_architecture.mul_single import build_cnn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

def get_pixels(img, width, height, channels):
	ret = []
	img = img.resize((width, height))
	pixels = list(img.getdata())
	ret.append([[[pixels[k*width+i][j] for k in range(height)] for i in range(width)] for j in range(channels)]) # (channel,width,height) shape
	ret = np.array(ret)
	ret = ret.astype('float32')
	return ret 


def sliding_window(input_path, model, model_img_width, model_img_height, cha_set):
	img = Image.open(input_path)
	img_width, img_height = img.size
	# window_size = min(img_height, img_width) 
	window_size = 30
	window_stride = 1

	pos = 0
	probas = []
	while pos+window_size-1 < img_width:
		window_img = img.crop((pos,0,pos+window_size-1,img_height))
		pixels = get_pixels(window_img, model_img_width, model_img_height, 3)
		probas.append(model.predict({'input':pixels}, verbose=0))
		for i in range(len(cha_set)):
			if probas[-1]['output%d'%(i+1)] > 0.9:
				print cha_set[i], pos, 
		print
		pos += window_stride
		window_img.save('crop_img/%d.jpg'%pos)
	for idx in range(len(cha_set)):
		cha = cha_set[idx]
		plt.title(cha)
		plt.plot([i['output%d'%(idx+1)][0][0] for i in probas], 'b')
		plt.savefig('prob_pic/'+cha+'.jpg')
		plt.close('all')




if __name__ == '__main__':
	img_width, img_height = 50, 50
	img_channels = 3 
	max_nb_cha = 1
	len_set = range(1, max_nb_cha+1)
	cha_set = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	nb_classes = 36
	test_data_dir = 'test_data/beijing/'
	weights_file_path = 'model/2016-05-30/weights.44-2.09.hdf5'
	
	model = build_cnn(img_channels, img_width, img_height, max_nb_cha, nb_classes)
	model.load_weights(weights_file_path)
	for i in range(1,2):
		sliding_window(test_data_dir+'%d.jpg'%i, model, img_width, img_height, cha_set)
		print 
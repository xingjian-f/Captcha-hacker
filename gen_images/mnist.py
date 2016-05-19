from keras.datasets import mnist
from PIL import Image
import os
import random

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# save_dir = 'img_data/train/mnist_resize/'
# os.mkdir(save_dir)
# cnt = 0
# with open(save_dir+'label.txt', 'w') as f:
# 	for i in range(len(X_train)):
# 		a = Image.fromarray(X_train[i])
# 		a = a.resize((14, 28))
# 		for j in range(5):
# 			b = Image.new('L', (28,28))
# 			b.paste(a, (j*2,0))
# 			cnt += 1
# 			b.save(save_dir+'%d.jpg'%(cnt))
# 			f.write(str(y_train[i])+'\r\n')
# 		print i

# save_dir = 'img_data/test/mnist_resize/'
# os.mkdir(save_dir)
# cnt = 0
# with open(save_dir+'label.txt', 'w') as f:
# 	for i in range(len(X_test)):
# 		a = Image.fromarray(X_test[i])
# 		a = a.resize((14, 28))
# 		for j in range(5):
# 			b = Image.new('L', (28,28))
# 			b.paste(a, (j*2,0))
# 			cnt += 1
# 			b.save(save_dir+'%d.jpg'%(cnt))
# 			f.write(str(y_test[i])+'\r\n')
# 		print i

save_dir = 'img_data/conca/'
# os.mkdir(save_dir)

with open(save_dir+'label.txt', 'w') as f1:
	with open(save_dir+'label1.txt', 'w') as f2:
		for i in range(10000):
			idxa = random.randint(0, len(X_test)-1)
			idxb = random.randint(0, len(X_test)-1)
			a = Image.fromarray(X_test[idxa])
			b = Image.fromarray(X_test[idxb])
			c = Image.new('L', (56,28))
			c.paste(a, (0,0))
			c.paste(b, (28, 0))
			c = c.resize((28, 28))
			c.save(save_dir+'%d.jpg'%(i+1))
			f1.write(str(y_test[idxa])+'\r\n')
			f2.write(str(y_test[idxb])+'\r\n') 
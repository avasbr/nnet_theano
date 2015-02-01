import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from deepnet.common import nnettrain as nt
from deepnet import MultilayerNet as mln
import theano
import sys

def load_mnist(data_path):
	''' Loads the MNIST data from the base path '''

	train_img_path = '%s/train-images.idx3-ubyte'%data_path
	train_lbl_path = '%s/train-labels.idx1-ubyte'%data_path 
	test_img_path = '%s/t10k-images.idx3-ubyte'%data_path
	test_lbl_path = '%s/t10k-labels.idx1-ubyte'%data_path

	def encode_one_hot(y,m,k):
		y_one_hot = np.zeros((m,k))
		y_one_hot[range(m),y] = 1
		return y_one_hot
	
	# get the training data
	train_img = idx2numpy.convert_from_file(train_img_path)
	m,row,col = train_img.shape
	d = row*col
	X_tr = np.reshape(train_img,(m,d))/255.

	train_lbl = idx2numpy.convert_from_file(train_lbl_path)
	k = max(train_lbl)+1
	y_tr = encode_one_hot(train_lbl,m,k) 
	
	# set the data matrix for test
	test_img = idx2numpy.convert_from_file(test_img_path)
	m_te = test_img.shape[0]
	X_te = np.reshape(test_img,(m_te,d))/255. # test data matrix
	test_lbl = idx2numpy.convert_from_file(test_lbl_path) 
	y_te = encode_one_hot(test_lbl,m_te,k)

	return X_tr,y_tr,X_te,y_te

def main(argv):
	data_path = argv[1] # directory path that contains all the MNIST data
	pt_config_file = argv[2] # config file for pre-training
	ft_config_file = argv[3] # config file for fine-tuning

	# load data
	print 'Loading data...'
	X_tr,y_tr,X_te,y_te = load_mnist(data_path)
	
	# train a neural network
	print 'Pre-training...'
	pt_wts = nt.train_nnet(pt_config_file,X_tr,y_tr=y_tr)
	
	print 'Fine-tuning...'
	# test it
	print 'Performance on test set:'
	print 100*nnet.score(X_te,y_te),'%'

if __name__ == '__main__':
	main(sys.argv)
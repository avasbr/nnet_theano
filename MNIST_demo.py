import idx2numpy
import numpy as np
import nnettrain as nt
import MultilayerNet as mln

def load_mnist(base_path):
	''' Loads the MNIST data from the base path '''

	train_img_path = '%s/train-images.idx3-ubyte'%base_path
	train_lbl_path = '%s/train-labels.idx1-ubyte'%base_path 
	test_img_path = '%s/t10k-images.idx3-ubyte'%base_path
	test_lbl_path = '%s/t10k-labels.idx1-ubyte'%base_path

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

def train_nnet_mnist(base_path=None,config_file=None):
	''' Trains a neural network on the MNIST data '''

	print 'Loading data...'
	
	# assume a default 
	if base_path is None:
		base_path = '/home/g64892/datasets/MNIST'
	config_file = '/home/g64892/Desktop/nnet_theano/MNIST_config.ini'
	X_tr,y_tr,X_te,y_te = load_mnist(base_path)

	# Train a model with the same learning rate on the training set, test on the testing set:
	print 'Training...'
	d = X_tr.shape[1]
	k = y_tr.shape[1]

	mln_params = {'d':d,'k':k,'num_hid':[50],'activ':['sigmoid','softmax'],
	'loss_terms':['cross_entropy','dropout'],'L2_decay':0.0001,'input_p':0.2,'hidden_p':0.5}

	rmsprop_params = {'method':'RMSPROP','opt_type':'minibatch','num_epochs':100,'batch_size':128,'learn_rate':0.01,
	'rho':0.9,'max_norm':False,'c':15}

	nnet = mln.MultilayerNet(**mln_params)
	nnet.fit(X_tr,y_tr,**rmsprop_params)

	print 'Performance on test set:'
	print 100*nnet.score(X_te,y_te),'%'

def main(argv):
	if len(argv) 
	base_path = argv[1] # directory path that contains all the MNIST data
	config_file = argv[2] # config file which holds all the parameters
	
	# load data
	print 'Loading data...'
	X_tr,y_tr,X_te,y_te = load_mnist(base_path)
	
	# train a neural network
	print 'Training...'
	nnet = nt.train_nnet(X_tr,y_tr,config_file)

	# test it
	print 'Performance on test set:'
	print 100*nnet.score(X_te,y_te),'%'

if __name__ == '__main__':
	main()
from deepnet import MultilayerNet as mln
from deepnet.common import nnetact as na
from deepnet.common import nnetutils as nu
import numpy as np
from hyperopt import hp,fmin,tpe
from math import log

# this dataset isn't changing, so just hard-code these values
m = 1000
d = 40
k = 2

def load_london_dataset(base_path):
	''' loads the london kaggle dataset '''
	X_tr = np.genfromtxt('%s/train.csv'%base_path,delimiter=',')
	y_tr = np.genfromtxt('%s/trainLabels.csv'%base_path,delimiter=',')
	X_te = np.genfromtxt('%s/test.csv'%base_path,delimiter=',')

	def encode_one_hot(y):
		y_one_hot = np.zeros((m,k))
		y_one_hot[range(m),y] = 1
		return y_one_hot
	y_oh = encode_one_hot(np.asarray(y_tr,dtype='int32'))

	return X_tr,y_oh,X_te

def write_pred_to_csv(y_pred):
	''' writes the predictions to a file which can be submitted to kaggle '''
	pass

# load the dataset
print 'Loading the london dataset...'
base_path = '/home/bhargav/datasets/kaggle_data/london/dataset'
X,y,X_te = load_london_dataset(base_path)

def compute_split_loss(mln_params,optim_params):
	'''re-shuffles the training/validation everytime - this is important because otherwise, we'll 
	just be overfitting to the validation set'''

	X_tr,y_tr,X_val,y_val = nu.split_train_val(X,y,0.6)
	
	nnet = mln.MultilayerNet(**mln_params)
	nnet.fit(X_tr,y_tr,**optim_params)
	loss = float(nnet.compute_test_loss(X_val,y_val))

	return loss

def compute_cv_loss(mln_params,optim_params):
	''' Uses k-fold cross-validation to compute the average loss '''

	k = 10
	cv_splits = nu.split_k_fold_cross_val(X,y,k=k)
	loss = 0.
	for split in cv_splits:
		X_tr,y_tr,X_val,y_val = split
		nnet = mln.MultilayerNet(**mln_params)
		nnet.fit(X_tr,y_tr,**optim_params)
		loss += float(nnet.compute_test_loss(X_val,y_val))

	avg_loss = 1.*loss/k
	print avg_loss

	return avg_loss # average loss

def hyperopt_obj_fn(args):
	''' hyper-opt objective function '''
	num_hid,learn_rate,l1_decay,l2_decay,activs,scale_factor,init_method,num_epochs,rho = args
	
	# multilayer net parameters
	mln_params = {'d':d,'k':k,'num_hids':[num_hid],'activs':activs,
	'loss_terms':['cross_entropy','regularization'],'l2_decay':l2_decay,'l1_decay':l1_decay}

	rmsprop_params = {'init_method':init_method,'scale_factor':scale_factor,'optim_type':'minibatch',
	'optim_method':'RMSPROP','batch_size':900,'num_epochs':num_epochs,'learn_rate':learn_rate,'rho':rho}
	
	return compute_cv_loss(mln_params,rmsprop_params)

rmsprop_space = (
	hp.qloguniform('num_hid',log(10),log(1000),1),
	hp.loguniform('learn_rate',log(1e-4),log(10)),
	hp.choice('l1_decay',[0,hp.loguniform('l1',log(1e-6),2.)]),
	hp.choice('l2_decay',[0,hp.loguniform('l2',log(1e-6),2.)]),
	hp.choice('activs',[['sigmoid','softmax'],['reLU','softmax']]),
	hp.loguniform('scale_factor',log(1e-3),log(1)),
	hp.choice('init_method',['gauss','fan-io']),
	hp.qloguniform('num_epochs',log(10),log(1e3),1),
	hp.uniform('rho',1e-2,0.99)
)

best = fmin(hyperopt_obj_fn, rmsprop_space, algo=tpe.suggest,max_evals=100)
print best
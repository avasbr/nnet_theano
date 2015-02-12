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
base_path = '/home/avasbr/datasets/kaggle/london_dataset'
X,y,X_te = load_london_dataset(base_path)

def get_loss(mln_params,sgd_params):
	# re-shuffles the training/validation everytime - this is important because otherwise, we'll just be
	# overfitting to the validation set
	X_tr,y_tr,X_val,y_val = nu.split_train_val(X,y,0.6)
	nnet = mln.MultilayerNet(**mln_params)
	nnet.fit(X_tr,y_tr,**sgd_params)
	loss = float(nnet.compute_test_loss(X_val,y_val))

	return loss

def hyperopt_obj_fn(args):
	''' hyper-opt objective function '''
	num_hid,learn_rate,l2_decay,scale_factor,init_method,num_epochs = args
	
	# multilayer net parameters
	mln_params = {'d':d,'k':k,'num_hids':[num_hid],'activs':['sigmoid','softmax'],
	'loss_terms':['cross_entropy','regularization'],'l2_decay':l2_decay}

	sgd_params = {'init_method':'gauss','scale_factor':scale_factor,'optim_type':'minibatch',
	'optim_method':'SGD','batch_size':600,'num_epochs':num_epochs,'learn_rate':learn_rate}
	
	return get_loss(mln_params,sgd_params)

# define the hyperopt configuration space
space = (
	hp.qloguniform('num_hid',log(10),log(1000),1),
	hp.loguniform('learn_rate',log(1e-4),log(100)),
	hp.loguniform('l2_decay',log(1e-5),log(10)),
	hp.loguniform('scale_factor',log(1e-3),log(1)),
	hp.choice('init_method',['gauss','fan-io']),
	hp.qloguniform('num_epochs',log(10),log(1e3),1),
)

best = fmin(hyperopt_obj_fn, space, algo=tpe.suggest,max_evals=100)
print best









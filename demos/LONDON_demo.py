from deepnet import MultilayerNet as mln
from deepnet.common import nnetact as na
from deepnet.common import nnetutils as nu
import numpy as np

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

X_tr,y_tr,X_val,y_val = nu.split_train_val(X,y,0.6)

# neural network parameters
mln_params = {'d':d,'k':k,'num_hids':[150],'activs':['sigmoid','softmax'],
'loss_terms':['cross_entropy','regularization'],'l2_decay':0.01}

# optimization parameters
lbfgs_params = {'init_method':'gauss','scale_factor':0.1,'optim_type':'fullbatch',
'optim_method':'L-BFGS-B','num_epochs':500,'plotting':True}

sgd_params = {'init_method':'gauss','scale_factor':0.1,'optim_type':'minibatch',
'optim_method':'SGD','batch_size':600,'num_epochs':500,'learn_rate':0.1,'plotting':True}

print 'Fitting a neural network...'
nnet = mln.MultilayerNet(**mln_params)
nnet.fit(X_tr,y_tr,X_val=X_val,y_val=y_val,**sgd_params)

print 'Performance on validation set:'
print 100*nnet.score(X_val,y_val),'%'
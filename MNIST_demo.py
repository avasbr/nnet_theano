import idx2numpy
import numpy as np
import nnetutils as nu
import theano
import theano.tensor as T
import MultilayerNet as mln

print 'Loading data...'

train_img_path = '/home/avasbr/datasets/MNIST/train-images.idx3-ubyte'
train_lbl_path = '/home/avasbr/datasets/MNIST/train-labels.idx1-ubyte' 
test_img_path = '/home/avasbr/datasets/MNIST/t10k-images.idx3-ubyte'
test_lbl_path = '/home/avasbr/datasets/MNIST/t10k-labels.idx1-ubyte'

# define training and validation data
train_img = idx2numpy.convert_from_file(train_img_path)
m,row,col = train_img.shape
d = row*col
X = np.reshape(train_img,(m,d))/255.

train_lbl = idx2numpy.convert_from_file(train_lbl_path)
k = max(train_lbl)+1

y = np.zeros((m,k)) # 'one-hot' representation
for i,idx in enumerate(train_lbl):
	y[i,idx] = 1

# set the data matrix for test
test_img = idx2numpy.convert_from_file(test_img_path)
m_te = test_img.shape[0]
X_te = nu.floatX(np.reshape(test_img,(m_te,d))/255.) # test data matrix
test_lbl = nu.floatX(idx2numpy.convert_from_file(test_lbl_path)) 

y_te = np.zeros((m_te,k)) # 'one-hot' representation
for i,idx in enumerate(test_lbl):
	y_te[i,idx] = 1

validation_flag = False
if validation_flag:
	# using a validation set to find the best global learning rate - need to experiment with changing learning rate
	learn_rate_candidates = [0.001,0.01,1,10,100,1000,1e4] # global learning rates
	X_tr,y_tr,X_val,y_val = nu.split_train_val(X,y,0.8)
	X_tr = nu.floatX(X_tr)
	y_tr = nu.floatX(y_tr)
	X_val = nu.floatX(X_val)
	y_val = nu.floatX(y_val)

	accuracies = [0.0]*len(learn_rate_candidates)
	best_accuracy = 0.0
	best_learn_rate = 0.0
	for i,learn_rate in enumerate(learn_rate_candidates):
		print 'Processing learning_rate:',learn_rate,'...'
		# define a new network to retrain
		mln_params = {'d':d,'k':k,'n_hid':[50],'activ':[nu.sigmoid,nu.softmax],'cost_type':'cross_entropy',
		'dropout_flag':True,'input_p':0.2,'hidden_p':0.5}
		optim_params = {'method':'SGD','n_epochs':2000,'batch_size':100,'learn_rate':learn_rate}
		nnet = mln.MultilayerNet(**mln_params)
		nnet.fit(X_val,y_val,**optim_params)
		pred,acc = nnet.get_predict_fns()
		accuracy = acc(X_val,y_val)
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_learn_rate = learn_rate

	print 'Best accuracy:',best_accuracy
	print 'Best learning rate:',best_learn_rate

# Train a model with the same learning rate on the training set, test on the testing set:
print 'Training...'
mln_params = {'d':d,'k':k,'n_hid':[800,800],'activ':[nu.sigmoid,nu.sigmoid,nu.softmax],'cost_type':'cross_entropy',
'dropout_flag':True,'input_p':0.2,'hidden_p':0.5}
optim_params = {'method':'SGD','n_epochs':2000,'batch_size':100,'learn_rate':10}
nnet = mln.MultilayerNet(**mln_params)
nnet.fit(X,y,**optim_params)
pred,acc = nnet.get_predict_fns()
print 100*acc(X_te,y_te),'%'


# print 'Training...'

# mln_params = {'d':d,'k':k,'n_hid':[50],'activ':[nu.sigmoid,nu.softmax],'cost_type':'cross_entropy','dropout_flag':False,'input_p':0.2,'hidden_p':0.5}
# # optim_params = {'method':'SGD','n_iter':1000,'learn_rate':0.1}
# # optim_params = {'method':'SGD','n_epochs':1000,'batch_size':500,'learn_rate':0.13,'early_stopping':True,'patience':7000}
# optim_params = {'method':'SGD','n_epochs':100,'batch_size':1000,'learn_rate':0.13}
# nnet = mln.MultilayerNet(**mln_params)
# nnet.fit(X,y_oh,**optim_params)

# pred,acc = nnet.get_predict_fns()

# print 'Performance on test set:'
# print 'Accuracy:',100.*acc(X_te,y_te),'%'
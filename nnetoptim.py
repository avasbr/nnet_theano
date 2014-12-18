import numpy as np
import theano
import theano.tensor as T
import nnetutils as nu
import copy

def maxnorm_regularization(w,c):
	''' clamping function which restricts the L2 norm of a weight vector to be of length c, in the L2 sense '''
	l2n = T.sum(w**2,axis=0)
	w /= ((l2n > c**2)*T.sqrt(l2n) + (l2n < c**2)*1.)

def vanilla_gradient_descent(X_tr,y_tr,wts,bs,compute_loss,compute_grad,n_iter,learn_rate):
	''' Simple, fat-free, reduced-sugar, low-calorie vanilla gradient descent
	
	Parameters:
	-----------
	
	Returns:
	--------
	'''

	# compile the train function
	X = T.matrix('X') # typed, input variable
	y = T.matrix('y') # type, output variable
	
	loss = compute_loss(X,y,wts,bs)
	d_loss_d_wts,d_loss_d_bs = compute_grad(loss,wts,bs) # the loss and gradients
	
	updates = []
	for wt,b,d_loss_d_wt,d_loss_d_b in zip(wts,bs,d_loss_d_wts,dbs):
		updates.append((wt,wt-learn_rate*d_loss_d_wt))
		updates.append((b,b-learn_rate*d_loss_d_b))

	# compiles the training function which defines how the losss will be changed, etc
	train = theano.function(inputs=[X,y],updates=updates)
	
	# simple, fat-free, reduced sugar vanilla gradient descent
	for i in range(n_iter):
		train(X_tr,y_tr)

def rmsprop(X_tr,y_tr,batch_size,n_epochs=1000):
	pass

def minibatch_gradient_descent(X_tr,y_tr,wts,bs,compute_loss,compute_grad,batch_size=100,
	n_epochs=500,learn_rate=0.1,max_norm=False,c=5):
	''' Assuming all the data can fit in memory, runs mini-batch gradient descent with optional max-norm
	regularization. This tends to work well with dropout + rectified linear activation functions '''
	
	m = X_tr.shape[0] # total number of training instances
	n_batches = int(m/batch_size) # number of batches, based on batch size
	leftover = m-n_batches*batch_size # batch_size won't divide the data evenly, so get leftover
	epoch = 0
	
	X = T.matrix('X') # input variable
	y = T.matrix('y') # output variable
	
	loss = compute_loss(X,y,wts,bs) # loss
	d_loss_d_wts,d_loss_d_bs = compute_grad(loss,wts,bs) # gradient 

	updates = []
	for wt,b,d_loss_d_wt,d_loss_d_b in zip(wts,bs,d_loss_d_wts,d_loss_d_bs):
		
		wt_ = wt-learn_rate*d_loss_d_wt
		b_ = b-learn_rate*d_loss_d_b
		
		# constrains the norm to lie on a ball of radius c
		if max_norm:
			maxnorm_regularization(wt_,c)

		# compute the actual update
		updates.append((wt,wt_))
		updates.append((b,b_))
	
	# compiles the training and validation functions
	train = theano.function(inputs=[X,y],updates=updates,mode='FAST_RUN',allow_input_downcast=True) # training function
	evaluate = theano.function(inputs=[X,y],outputs=loss,mode='FAST_RUN',allow_input_downcast=True) # useful also for validation purposes
	
	while epoch < n_epochs:
		epoch += 1
		tr_idx = np.random.permutation(m) # randomly shuffle the data indices
		ss_idx = range(0,m,batch_size)
		ss_idx[-1] += leftover # add the leftovers to the last batch
		
		# run through a full epoch
		for idx,(start_idx,stop_idx) in enumerate(zip(ss_idx[:-1],ss_idx[1:])):			
			n_batch_iter = (epoch-1)*n_batches + idx # total number of batches processed up until now
			batch_idx = tr_idx[start_idx:stop_idx] # get the next batch
			train(X_tr[batch_idx,:],y_tr[batch_idx,:]) # update the model
			# uncomment the next two lines for speed - this is purely for reporting purposes, should be in a log
			# tr_loss = evaluate(X_tr,y_tr)
			# print 'iter/epoch: %i/%i, training loss: %.3f'%(idx,epoch,tr_loss)
		if epoch%10 == 0:
			tr_loss = evaluate(X_tr,y_tr)
			print 'Epoch: %s, Training error: %.3f'%(epoch,tr_loss)

def adagrad(X_tr,y_tr,n_nodes,wts,bs,compute_loss,compute_grad,batch_size=100,n_epochs=500,
	master_learn_rate=1.,max_norm=False,c=5):
	''' adaptive gradient method - typically works better than vanilla SGD and has some 
	nice theoretical guarantees

	Parameters:
	----------

	param: X_tr - training dataset
	type: theano matrix

	param: y_tr - training labels
	type: theano matrix
	
	param: n_nodes - number of nodes per layer (except the input)
	type: list of ints

	param: wts - weights which need to be optimized
	type: shared theano variable

	param: bs - bias weights
	type: shared theano variable

	param: compute_loss - computes the loss at current point
	type: function
	
	param: compute_grad - function to compute the loss and gradient at the current point
	type: function

	param: batch_size - number of examples per mini-batch
	type: int

	param: n_epochs - the number of full runs through the dataset
	type: int

	param: X_val - validation dataset
	type: theano matrix

	param: y_val - validation labels
	type: theano matrix 

	Returns:
	--------
	None

	Updates:
	--------
	wts,bs
	'''

	# compile the training function	
	m = X_tr.shape[0] # total number of training instances
	n_batches = int(m/batch_size) # number of batches, based on batch size
	leftover = m-n_batches*batch_size # batch_size won't divide the data evenly, so get leftover
	epoch = 0
	eps = 1e-7

	X = T.matrix('X') # input variable
	y = T.matrix('y') # output variable
	
	loss = compute_loss(X,y,wts,bs) # loss
	d_loss_d_wts,d_loss_d_bs = compute_grad(loss,wts,bs) # gradient 
	
	# initialize the historical gradient parameters
	h_d_loss_d_wts = [None]*(len(n_nodes)-1); h_d_loss_d_bs = [None]*(len(n_nodes)-1)
	
	for i,(n1,n2) in enumerate(zip(n_nodes[:-1],n_nodes[1:])):
		h_d_loss_d_wts[i] = nu.floatX(np.zeros((n1,n2)))
		h_d_loss_d_bs[i] = nu.floatX(np.zeros(n2))

	updates = []
	
	for wt,b,d_loss_d_wt,d_loss_d_b,h_d_loss_d_wt,h_d_loss_d_b in \
		zip(wts,bs,d_loss_d_wts,d_loss_d_bs,h_d_loss_d_wts,h_d_loss_d_bs):
		
		# historical gradient
		h_d_loss_d_wt += d_loss_d_wt**2
		h_d_loss_d_b += d_loss_d_b**2
		
		# update rule
		wt_ = wt - master_learn_rate*d_loss_d_wt/(T.sqrt(h_d_loss_d_wt) + eps)
		b_ = b - master_learn_rate*d_loss_d_b/(T.sqrt(h_d_loss_d_b) + eps)

		# constrains the norm to lie on a ball of radius c
		if max_norm:
			maxnorm_regularization(wt_,c)
		
		# compute the actual update
		updates.append((wt,wt_))
		updates.append((b,b_))

	train = theano.function(inputs=[X,y],updates=updates,mode='FAST_RUN',allow_input_downcast=True) # training function
	evaluate = theano.function(inputs=[X,y],outputs=loss,mode='FAST_RUN',allow_input_downcast=True) # useful also for validation purposes
	
	while epoch < n_epochs:
		epoch += 1
		tr_idx = np.random.permutation(m) # randomly shuffle the data indices
		ss_idx = range(0,m,batch_size)
		ss_idx[-1] += leftover # add the leftovers to the last batch
		
		# run through a full epoch
		for idx,(start_idx,stop_idx) in enumerate(zip(ss_idx[:-1],ss_idx[1:])):			
			n_batch_iter = (epoch-1)*n_batches + idx # total number of batches processed up until now
			batch_idx = tr_idx[start_idx:stop_idx] # get the next batch
			train(X_tr[batch_idx,:],y_tr[batch_idx,:]) # update the model
			# uncomment the next two lines for speed - this is purely for reporting purposes, should be in a log
			# tr_loss = evaluate(X_tr,y_tr)
			# print 'iter/epoch: %i/%i, training loss: %.3f'%(idx,epoch,tr_loss)
		if epoch%10 == 0:
			tr_loss = evaluate(X_tr,y_tr)
			print 'Epoch: %s, Training error: %.3f'%(epoch,tr_loss)

# Save for later
#--------
# def minibatch_gradient_descent_early_stopping(X_tr,y_tr,wts,bs,compute_loss_grad,compute_loss,batch_size=None,
# 	n_epochs=None,learn_rate=None,early_stopping=False,X_val=None,y_val=None,patience=None,
# 	patience_increase=2,improvement_threshold=0.995):
# 	''' Assuming all the data can fit in memory, runs mini-batch gradient descent with optional 
# 	early stopping (default False). The total number of iterations is controlled by the batch 
# 	size and the number of epochs.

# 	Parameters:
# 	----------
# 	param: wts - weights which need to be optimized
# 	type: shared theano variable

# 	param: bs - bias weights
# 	type: shared theano variable

# 	param: X_tr - training dataset
# 	type: theano matrix

# 	param: y_tr - training labels
# 	type: theano matrix

# 	param: batch_size - number of examples per mini-batch
# 	type: int

# 	param: compute_loss_grad - function to compute the loss and gradient at the current point
# 	type: function

# 	param: compute_loss - only computes the loss at current point
# 	type: function

# 	param: n_epochs - the number of full runs through the dataset
# 	type: int

# 	param: early_stopping - simple method to prevent overfitting
# 	type: boolean

# 	param: X_val - validation dataset
# 	type: theano matrix

# 	param: y_val - validation labels
# 	type: theano matrix 

# 	param: patience - number of batches to process before stopping
# 	type: int

# 	param: patience_increase - multiplicative factor to increase patience threshold
# 	type: int

# 	param: improvement_threshold - factor of improvement which we deem significant
# 	type: float

# 	Returns:
# 	--------
# 	None

# 	Updates:
# 	--------
# 	wts,bs

# 	'''
# 	m = X_tr.shape[0] # total number of training instances
# 	print 'Number of training instances:',m
# 	n_batches = int(m/batch_size) # number of batches, based on batch size
# 	print 'Number of batches:',n_batches
# 	leftover = m-n_batches*batch_size # batch_size won't divide the data evenly, so get leftover
# 	epoch = 0
# 	done_looping = False
# 	validation_frequency = min(n_batches,patience/2) # ensures that we validate at least twice before stopping
	
# 	x = T.matrix('x') # typed, input variable
# 	y = T.matrix('y') # type, output variable
	
# 	dummy,dW,db = compute_loss_grad(x,y,wts,bs) # the loss and gradients
# 	loss = compute_loss(x,y,wts,bs)

# 	updates = []
# 	for w,b,gw,gb in zip(wts,bs,dW,db):
# 		updates.append((w,w-learn_rate*gw))
# 		updates.append((b,b-learn_rate*gb))

# 	# compiles the training and validation functions
# 	train = theano.function(inputs=[x,y],updates=updates) # training function
# 	validate = theano.function(inputs=[x,y],outputs=loss) # validation functio

# 	best_val_loss = np.inf
# 	best_wts = None
# 	best_bs = None

# 	while epoch < n_epochs and not done_looping:
# 		epoch += 1
# 		tr_idx = np.random.permutation(m) # randomly shuffle the data indices
# 		ss_idx = range(0,m,batch_size)
# 		ss_idx[-1] += leftover # add the leftovers to the last batch
		
# 		# run through a full epoch
# 		for idx,(start_idx,stop_idx) in enumerate(zip(ss_idx[:-1],ss_idx[1:])):
			
# 			n_batch_iter = (epoch-1)*n_batches + idx # total number of batches processed up until now
# 			batch_idx = tr_idx[start_idx:stop_idx] # get the next batch
# 			train(X_tr[batch_idx,:],y_tr[batch_idx,:]) # update the model

# 			# we only worry about validation if early stopping is set to true
# 			if early_stopping:
				
# 				if n_batch_iter % validation_frequency == 0: # time to check performance on validation set
					
# 					val_loss = validate(X_val,y_val)
# 					print 'Batch iteration:',n_batch_iter,' Validation loss:',val_loss,' Patience:',patience
# 					if val_loss < best_val_loss:
# 						if val_loss < best_val_loss*improvement_threshold: # this is a significant improvement
# 							patience = max(patience,n_batch_iter*patience_increase) # increase patience, if possible
						
# 						best_val_loss = val_loss
# 						best_wts = copy.deepcopy(wts)
# 						best_bs = copy.deepcopy(bs)

# 				if n_batch_iter > patience:
# 					done_looping = True
# 					break # break out of the for-loop, and finish early_stopping
	
# 	if early_stopping:
# 		# we want to use these for our model
# 		wts = best_wts
# 		bs = best_bs
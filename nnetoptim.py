import numpy as np
import theano
import theano.tensor as T
import copy

def maxnorm_regularization(w,c):
	''' clamping function which restricts the L2 norm of a weight vector to be of length c, in the L2 sense '''
	l2n = T.sum(w**2,axis=0)
	w /= ((l2n > c**2)*T.sqrt(l2n) + (l2n < c**2)*1.)

def gradient_descent(X_tr,y_tr,wts,bs,compute_cost_grad,n_iter=1000,learn_rate=None):
	''' Simple, fat-free, reduced-sugar, low-calorie vanilla gradient descent
	
	Parameters:
	-----------
	
	Returns:
	--------
	'''

	# compile the train function
	x = T.matrix('x') # typed, input variable
	y = T.matrix('y') # type, output variable
	
	cost,dW,db = compute_cost_grad(x,y,wts,bs) # the cost and gradients
	
	updates = []
	for w,b,gw,gb in zip(wts,bs,dW,db):
		updates.append((w,w-learn_rate*gw))
		updates.append((b,b-learn_rate*gb))

	# compiles the training function which defines how the costs will be changed, etc
	train = theano.function(inputs=[x,y],updates=updates)
	
	# simple, fat-free, reduced sugar vanilla gradient descent
	for i in range(n_iter):
		train(X_tr,y_tr)

def rmsprop(X_tr,y_tr,batch_size,n_epochs=1000):
	pass

def minibatch_gradient_descent(X_tr,y_tr,wts,bs,compute_cost,compute_grad,batch_size=None,n_epochs=None,learn_rate=None,
	max_norm=False,c=None):
	''' Assuming all the data can fit in memory, runs mini-batch gradient descent with optional max-norm
	regularization. This tends to work well with dropout + rectified linear activation functions '''
	
	m = X_tr.shape[0] # total number of training instances
	n_batches = int(m/batch_size) # number of batches, based on batch size
	leftover = m-n_batches*batch_size # batch_size won't divide the data evenly, so get leftover
	epoch = 0
	
	x = T.matrix('x') # input variable
	y = T.matrix('y') # output variable
	
	cost = compute_cost(x,y,wts,bs) # cost
	dW,db = compute_grad(cost,wts,bs) # gradient 

	updates = []
	for w,b,gw,gb in zip(wts,bs,dW,db):
		w_ = w-learn_rate*gw
		b_ = b-learn_rate*gb
		# constrains the norm to lie on a ball of radius c
		if max_norm:
			maxnorm_regularization(w_,c)
		# compute the actual update
		updates.append((w,w_))
		updates.append((b,b_))
	
	# compiles the training and validation functions
	train = theano.function(inputs=[x,y],updates=updates) # training function
	evaluate = theano.function(inputs=[x,y],outputs=cost) # useful also for validation purposes
	while epoch < n_epochs:
		print 'epoch',epoch
		epoch += 1
		tr_idx = np.random.permutation(m) # randomly shuffle the data indices
		ss_idx = range(0,m,batch_size)
		ss_idx[-1] += leftover # add the leftovers to the last batch
		
		# run through a full epoch
		for idx,(start_idx,stop_idx) in enumerate(zip(ss_idx[:-1],ss_idx[1:])):			
			n_batch_iter = (epoch-1)*n_batches + idx # total number of batches processed up until now
			batch_idx = tr_idx[start_idx:stop_idx] # get the next batch
			train(X_tr[batch_idx,:],y_tr[batch_idx,:]) # update the model
			# print 'iter/epoch: %i/%i'%(idx,epoch)			
			# uncomment the next two lines for speed - this is purely for reporting purposes, should be in a log
			# tr_cost = evaluate(X_tr,y_tr)
			# print 'iter/epoch: %i/%i, training loss: %.3f'%(idx,epoch,tr_cost)
		if epoch%100 == 0:
			tr_cost = evaluate(X_tr,y_tr)
			print 'Epoch: %s, Training error:%.3f'%(epoch,tr_cost)

# Save for later
#--------
# def minibatch_gradient_descent_early_stopping(X_tr,y_tr,wts,bs,compute_cost_grad,compute_cost,batch_size=None,
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

# 	param: compute_cost_grad - function to compute the cost and gradient at the current point
# 	type: function

# 	param: compute_cost - only computes the cost at current point
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
	
# 	dummy,dW,db = compute_cost_grad(x,y,wts,bs) # the cost and gradients
# 	cost = compute_cost(x,y,wts,bs)

# 	updates = []
# 	for w,b,gw,gb in zip(wts,bs,dW,db):
# 		updates.append((w,w-learn_rate*gw))
# 		updates.append((b,b-learn_rate*gb))

# 	# compiles the training and validation functions
# 	train = theano.function(inputs=[x,y],updates=updates) # training function
# 	validate = theano.function(inputs=[x,y],outputs=cost) # validation functio

# 	best_val_cost = np.inf
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
					
# 					val_cost = validate(X_val,y_val)
# 					print 'Batch iteration:',n_batch_iter,' Validation cost:',val_cost,' Patience:',patience
# 					if val_cost < best_val_cost:
# 						if val_cost < best_val_cost*improvement_threshold: # this is a significant improvement
# 							patience = max(patience,n_batch_iter*patience_increase) # increase patience, if possible
						
# 						best_val_cost = val_cost
# 						best_wts = copy.deepcopy(wts)
# 						best_bs = copy.deepcopy(bs)

# 				if n_batch_iter > patience:
# 					done_looping = True
# 					break # break out of the for-loop, and finish early_stopping
	
# 	if early_stopping:
# 		# we want to use these for our model
# 		wts = best_wts
# 		bs = best_bs
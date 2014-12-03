import numpy as np
import theano
import theano.tensor as T

def gradient_descent(X_tr,y_tr,wts,bs,compute_cost,n_iter=1000,learn_rate=None):
	''' Simple, fat-free, reduced-sugar, low-calorie vanilla gradient descent
	
	Parameters:
	-----------
	
	Returns:
	--------
	'''

	# compile the train function
	x = T.matrix('x') # typed, input variable
	y = T.matrix('y') # type, output variable
	
	cost = compute_cost(x,y,wts,bs) # the cost and gradients
	
	# autodiff to compute gradient (essentially implements bprop)
	grads = T.grad(cost, [p for sublist in [wts,bs] for p in sublist])
	dW = grads[:len(wts)] # collect gradients for weight matrices...
	db = grads[len(wts):]# ...and biases
	
	updates = []
	for w,b,gw,gb in zip(wts,bs,dW,db):
		updates.append((w,w-learn_rate*gw))
		updates.append((b,b-learn_rate*gb))

	# compiles the training function which defines how the costs will be changed, etc
	train = theano.function(inputs=[x,y],outputs=cost,updates=updates)
	
	# simple, fat-free, reduced sugar vanilla gradient descent
	for i in range(n_iter):
		train(X_tr,y_tr)
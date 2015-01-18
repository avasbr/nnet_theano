import unittest
import numpy as np
import MultilayerNet as mln
import Autoencoder as ae
import nnetact as na
import nnetutils as nu
import theano
import theano.tensor as T
import copy
# from theano.tensor.shared_randomstreams import RandomStreams
# srng = RandomStreams()

def sigmoid_grad(z):
	return z*(1-z)

def sigmoid(z):
	return 1./(1+np.exp(-1.*z))

def g_x(xi,yi,wts):
	return 2*(xi-yi)*sigmoid_grad(wts[2]*sigmoid(wts[0]*xi) \
		+ wts[3]*sigmoid(wts[1]*xi))

class testOptim(unittest.TestCase):
	
	def testSGD(self):
		# toy data, and simulation of simple stochastic gradient descent
		X_data = [-5,-1,1,2,3]
		y_data = [25,1,1,4,9]
		wts = [0.01,-0.01,-0.02,0.005]
		params = [theano.shared(nu.floatX([[wts[0],wts[1]]])),theano.shared(nu.floatX([[wts[2]],[wts[3]]]))]
		grads = [0.,0.,0.,0.]
		learn_rate = 0.0005
		num_iter = 10
		N = len(X)
		
		for n in range(num_iter):
			
			# gradients	
			grads[0] = 2./N*sum([g_x(xi,yi,wts)*wts[2]*sigmoid_grad(wts[0]*xi) for xi,yi in zip(X_data,y_data)])
			grads[1] = 2./N*sum([g_x(xi,yi,wts)*wts[3]*sigmoid_grad(wts[1]*xi) for xi,yi in zip(X_data,y_data)])
			grads[2] = 2./N*sum([g_x(xi,yi,wts)*sigmoid(wts[0]*xi) for xi,yi in zip(X_data,y_data)])
			grads[3] = 2./N*sum([g_x(xi,yi,wts)*sigmoid(wts[1]*xi) for xi,yi in zip(X_data,y_data)])

			# weight updates
			wts = [wt-learn_rate*grad for wt,grad in zip(wts,grads)]
		print wts

		# # construct the expression graph for this simple network
		X = T.matrix()
		y = T.vector()
		
		y_pred = na.sigmoid(T.dot(na.sigmoid(T.dot(Xt,wts_shared[0])),wts_shared[1]))
		loss = nl.squared_error(y,y_pred)
		grads = [T.grad(loss,param) for param in params]
		updates = nopt.sgd(params,grads,learn_rate=learn_rate)
		train = theano.function(inputs=[X,y],updates=updates)

		for n in range(num_iter):
			train(X_data,y_data)

def main():
	unittest.main()

if __name__ == '__main__':
	main()


import unittest
import numpy as np
import MultilayerNet as mln
import Autoencoder as ae
import nnetutils as nu
import theano
import theano.tensor as T
import copy
# from theano.tensor.shared_randomstreams import RandomStreams
# srng = RandomStreams()

class testNeuralNetworkCore(unittest.TestCase):

	def setUp(self):
		
		# # initialize some random data and labels
		self.d = 5
		self.k = 3
		self.m = 100

		self.X = np.random.rand(self.m,self.d) # generate some synthetic data (5-dim feature vectors)
		self.Y = np.zeros((self.m,self.k))
		for i,v in enumerate(np.random.randint(0,self.k,self.m)):
			self.Y[i,v] = 1 # create one-hot vectors
		
		# define some gradient checking parameters
		self.err_tol = 1e-9
		self.eps = 1e-4

	def check_gradients(self,nnet,X_in,Y_in):
		
		X = T.matrix() # inputs
		Y = T.matrix() # labels
		v = T.vector() # vector of biases and weights
		i = T.lscalar() # index

		# 1. compile the numerical gradient function
		def compute_numerical_gradient(v,i,X,Y):
			
			# perturb the input
			v_plus = T.inc_subtensor(v[i],self.eps)
			v_minus = T.inc_subtensor(v[i],-1.0*self.eps)

			# roll it back into the weight matrices and bias vectors
			wts_plus, bs_plus = nu.t_reroll(v_plus,nnet.n_nodes)
			wts_minus, bs_minus = nu.t_reroll(v_minus,nnet.n_nodes)
			
			# compute the cost for both sides, and then compute the numerical gradient
			cost_plus = nnet.compute_cost(X,Y,wts_plus,bs_plus)
			cost_minus = nnet.compute_cost(X,Y,wts_minus,bs_minus)
			
			return 1.0*(cost_plus-cost_minus)/(2*self.eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)

		compute_ngrad = theano.function(inputs=[v,i,X,Y],outputs=compute_numerical_gradient(v,i,X,Y))

		# 2. compile backprop (theano's autodiff)
		cost = nnet.compute_cost(X,Y)
		dW,db = nnet.compute_grad(cost)
		compute_bgrad = theano.function(inputs=[X,Y],outputs=nu.t_unroll(dW,db))

		# compute the mean difference between the numerical and exact gradients
		v0 = nu.unroll([w.get_value() for w in nnet.wts_],[b.get_value() for b in nnet.bs_])
		n = np.size(v0)
		idxs = np.random.permutation(n)[:(n/1)] # get the indices of the weights/biases we want to check

 		ngrad = [None]*len(idxs)
		for j,idx in enumerate(idxs):
			ngrad[j] = compute_ngrad(v0,idx,X_in,Y_in)
		bgrad = compute_bgrad(X_in,Y_in)[idxs]

		cerr = np.mean(np.abs(ngrad-bgrad))
		self.assertLess(cerr,self.err_tol)

	def test_Autoencoder(self):
		'''  '''
		ae_params = {'d':self.d,'n_hid':50,'activ':[nu.sigmoid,nu.softmax],
		'cost_type':'cross_entropy','beta':3,'rho':0.1,'L2_decay':0.1}
		nnet = ae.Autoencoder(**ae_params)
				
		self.check_gradients(nnet,self.X,self.X)

	def test_mln_squared_error(self):
		mln_params = {'d':self.d,'k':self.k,'n_hid':[50],'activ':[nu.sigmoid,nu.softmax],
		'cost_type':'squared_error','L1_decay':0.1}
		nnet = mln.MultilayerNet(**mln_params)

		self.check_gradients(nnet,self.X,self.Y)

	def test_mln_singlelayer(self):

		mln_params = {'d':self.d,'k':self.k,'n_hid':[],'activ':[nu.softmax],
		'cost_type':'cross_entropy','L2_decay':0.1}
		nnet = mln.MultilayerNet(**mln_params)
		
		self.check_gradients(nnet,self.X,self.Y)
	
	def test_mln_multilayer(self):

		mln_params = {'d':self.d,'k':self.k,'n_hid':[50,50],'activ':[nu.sigmoid,nu.sigmoid,nu.softmax],
		'cost_type':'cross_entropy','L2_decay':0.1}
		nnet = mln.MultilayerNet(**mln_params)
				
		self.check_gradients(nnet,self.X,self.Y)

def main():
	unittest.main()

if __name__ == '__main__':
	main()
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
		self.y = np.zeros((self.m,self.k))
		for i,v in enumerate(np.random.randint(0,self.k,self.m)):
			self.y[i,v] = 1 # create one-hot vectors
		
		# define some gradient checking parameters
		self.err_tol = 1e-9
		self.eps = 1e-4

	def check_gradients(self,nnet):
		
		X = T.matrix() # inputs
		Y = T.matrix() # labels
		v_plus = T.vector() # full 'plus' weight vector
		v_minus = T.vector() # full 'minus' weight vector
		i = T.lscalar() # index

		# 1. compile the numerical gradient function
		def compute_numerical_gradient(v_plus,v_minus,X,Y):
			
			wts_plus, bs_plus = nu.theano_reroll(v_plus,nnet.n_nodes)
			wts_minus, bs_minus = nu.theano_reroll(v_minus,nnet.n_nodes)
			
			# compute the cost for both sides, and then compute the numerical gradient
			cost_plus = nnet.compute_cost(X,Y,wts_plus,bs_plus)
			cost_minus = nnet.compute_cost(X,Y,wts_minus,bs_minus)
			
			return 1.0*(cost_plus-cost_minus)/(2*self.eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)

		compute_ngrad = theano.function(inputs=[v_plus,v_minus,X,Y],outputs=compute_numerical_gradient(v_plus,v_minus,X,Y))

		# 2. compile backprop (theano's autodiff)
		cost = nnet.compute_cost(X,Y)
		dW,db = nnet.compute_grad(cost)
		compute_bgrad = theano.function(inputs=[X,Y],outputs=nu.theano_unroll(dW,db))

		# compute the mean difference
		v0 = nu.unroll([w.get_value() for w in nnet.wts_],[b.get_value() for b in nnet.bs_])
		n = np.size(v0)
		idxs = np.random.permutation(n)[:(n/1)] # choose a random 20% - we don't need to add this into the theano graph

 		# compute gradients
 		ngrad = [None]*len(idxs)
		for j,idx in enumerate(idxs):
			v_plus = copy.deepcopy(v0)
			v_minus = copy.deepcopy(v0)
			
			v_plus[idx] += self.eps
			v_minus[idx] -= self.eps
			
			ngrad[j] = compute_ngrad(v_plus,v_minus,self.X,self.y)

		bgrad = compute_bgrad(self.X,self.y)[idxs]

		# compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad))
		self.assertLess(cerr,self.err_tol)

	# def test_Autoencoder(self):

	# 	n_hid = 50
	# 	nnet = ae.Autoencoder(d=self.d,n_hid=n_hid)
				
	# 	wts = []
	# 	bs = []

	# 	for n1,n2 in zip(nnet.n_nodes[:-1],nnet.n_nodes[1:]):
	# 		wts.append(np.random.randn(n2,n1))
	# 		bs.append(np.random.randn(n2,1))

	# 	self.gradient_checker(wts,bs,self.X,self.X,nnet)

	def test_mln_single_layer(self):

		mln_params = {'d':self.d,'k':self.k,'n_hid':[50],'activ':[nu.sigmoid,nu.softmax],
		'cost_type':'cross_entropy','L2_decay':0.1}
		nnet = mln.MultilayerNet(**mln_params)
				
		self.check_gradients(nnet)
	
	# def test_softmax_multilayer(self):
	# 	''' Gradient checking of backprop for multi-hidden-layer softmax '''

	# 	n_hid = [50,25]
	# 	decay = 0.2
	# 	nnet = scl.SoftmaxClassifier(d=self.d,k=self.k,n_hid=n_hid,decay=decay)
		
	# 	wts = []
	# 	bs = []

	# 	for n1,n2 in zip(nnet.n_nodes[:-1],nnet.n_nodes[1:]):
	# 		wts.append(np.random.randn(n2,n1))
	# 		bs.append(np.random.randn(n2,1))

	# 	self.gradient_checker(wts,bs,self.X,self.y,nnet)

def main():
	unittest.main()

if __name__ == '__main__':
	main()
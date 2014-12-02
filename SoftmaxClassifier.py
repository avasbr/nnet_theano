import numpy as np
import NeuralNetworkCore
import theano
import theano.tensor as T

class SoftmaxClassifier(NeuralNetworkCore.Network):

	def __init__(self,d=None,k=None,n_hid=None,activ=None,cost_type='cross_entropy',**hyperparam):
		
		''' simply calls the superclass constructor with the appropriate cost function'''
		
		if cost_type == 'cross_entropy':
			NeuralNetworkCore.Network.__init__(self,d=d,k=k,n_hid=n_hid,cost_func=self.cross_entropy,**hyperparam)
		elif cost_type == 'squared_error':
			NeuralNetworkCore.Network.__init__(self,d=d,k=k,n_hid=n_hid,cost_func=self.squared_error,**hyperparam)
		else:
			sys.exit("That cost function is not available")

	def cross_entropy(self,y,wts=None,bs=None,**hyperparam):
		''' basic cross entropy cost function '''
		
		reg_cost = 0

		# usually there would be just one or the other.. not sure when you would use both,
		# seems like they would compete		
		if 'L1_decay' in hyperparam:
			reg_cost = hyperparam['L1_decay']*sum([T.sum(T.abs(w)) for w in wts])
		
		if 'L2' in hyperparam:
			reg_cost = 0.5*hyperparam['L2_decay']*sum([T.sum(w**2) for w in wts])

		E = T.mean(T.sum(-1.0*y*T.log(self.act[-1]),axis=0)) + reg_cost

		return E


	def squared_error(self,y,wts=None,bs=None,**hyperparam):
		''' basic squared error cost function '''

		reg_cost = 0		

		# usually there would be just one or the other.. not sure when you would use both,
		# seems like they would compete
		if 'L1_decay' in hyperparam:
			reg_cost += hyperparam['L1_decay']*sum([T.sum(T.abs(w)) for w in wts])

		if 'L2_decay' in hyperparam:
			reg_cost += 0.5*hyperparam['L2_decay']*sum([T.sum(w**2) for w in wts])

		E = T.mean(T.sum((y-self.act[-1])**2)) + reg_cost

		return E




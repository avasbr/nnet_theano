import numpy as np
import NeuralNetworkCore
import theano
import theano.tensor as T

class MultilayerNet(NeuralNetworkCore.Network):

	def __init__(self,d=None,k=None,n_hid=None,activ=None,cost_type='cross_entropy',**cost_params):
		''' simply calls the superclass constructor with the appropriate cost function'''
		
		if cost_type == 'cross_entropy':
			NeuralNetworkCore.Network.__init__(self,d=d,k=k,n_hid=n_hid,activ=activ,cost_func=self.cross_entropy,**cost_params)
		
		elif cost_type == 'squared_error':
			NeuralNetworkCore.Network.__init__(self,d=d,k=k,n_hid=n_hid,activ=activ,cost_func=self.squared_error,**cost_params)
		
		else:
			sys.exit("That cost function is not available")

	def cross_entropy(self,y,wts=None,bs=None,**cost_params):
		''' basic cross entropy cost function with optional regularization'''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_
		
		reg_cost = 0

		# usually there would be just one or the other.. not sure when you would use both,
		# seems like they would compete
		if 'L1_decay' in cost_params:
			reg_cost = cost_params['L1_decay']*sum([T.sum(T.abs(w)) for w in wts])
		
		if 'L2_decay' in cost_params:
			reg_cost = 0.5*cost_params['L2_decay']*sum([T.sum(w**2) for w in wts])

		E = T.mean(T.sum(-1.0*y*T.log(self.act[-1]),axis=1)) + reg_cost

		return E

	def squared_error(self,y,wts=None,bs=None,**cost_params):
		''' basic squared error cost function with optional regularization'''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_
		
		reg_cost = 0		

		# usually there would be just one or the other.. not sure when you would use both,
		# seems like they would compete
		if 'L1_decay' in cost_params:
			reg_cost += cost_params['L1_decay']*sum([T.sum(T.abs(w)) for w in wts])

		if 'L2_decay' in cost_params:
			reg_cost += 0.5*cost_params['L2_decay']*sum([T.sum(w**2) for w in wts])

		E = T.mean(T.sum((y-self.act[-1])**2)) + reg_cost

		return E

	def get_predict_fns(self,wts=None,bs=None):
		''' predicts the class of the input, and additionally the misclassification error if the
		true labels are provided'''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		X = T.matrix()
		y = T.vector()
		self.fprop(X,wts,bs)
		pred = T.argmax(self.act[-1],axis=1)

		# compile the functions - this is what the user can use to do prediction
		pred_func = theano.function([X],pred)
		mce_func = theano.function([X,y],1.0-T.mean(T.neq(pred,y)))

		return pred_func,mce_func
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

	def get_predict_fns(self,wts=None,bs=None):
		'''This might be slightly confusing, as this function returns two compiled functions which can 
		be used for testing purposes. Since theano needs to first compile expressions to actually evaluate
		them, it makes sense to return their compiled versions such that a user can then freely use them 
		to perform classification. Otherwise, we would be recompiling the expressions with every call, which
		is definitely non-ideal'''

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

	
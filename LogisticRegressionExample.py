 Logistic regression
import numpy
from theano import function, shared
import theano.tensor as T
rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N,feats),rng.randint(size=N,low=0,high=2)) # data
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = shared(rng.randn(feats),name="w") # weight vector
b = shared(0.,name="b") # bias vector

print "Intial model:"
print w.get_value(), b.get_value()

# Construct Theano expression graph - this essentially defines the cost function 
# and gradient
print "Building Theano expression graph..."
decay = 0.01
learn_rate = 0.1
p_1 = 1./(1. + T.exp(-T.dot(x,w)-b)) # probability that target = 1 (sigmoid)
pred_func = p_1 > 0.5 # prediction thresholded
x_ent = -y*T.log(p_1) - (1-y)*T.log(1-p_1) # cross-entropy loss-function
cost = x_ent.mean() + decay*(w**2).sum() # regularized cost function
gw,gb = T.grad(cost,[w,b]) # gradients via autodiff, with respect to w,b

# Compile - the main function is one iterative step in training.
# The 'updates' parameter in this context is our optimization routine
print "Compiling.." 
train = function([x,y],[pred_func,x_ent],updates=((w,w-learn_rate*gw),(b,b-learn_rate*gb)))
predict = function([x],pred_func)

# Train
for i in range(training_steps):
	pred,err = train(D[0],D[1])

print "Final model:"
print w.get_value(), b.get_value()
print "Target values for D:",D[1]
print "Prediction on D:", predict(D[0])
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print f.maker.fgraph.toposort()
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print 'Looping %d times took' % iters, t1 - t0, 'seconds'
print 'Result is', r
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print 'Used the cpu'
else:
    print 'Used the gpu'

# Results:

# $ THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python check1.py
# [Elemwise{exp,no_inplace}(<TensorType(float32, vector)>)]
# Looping 1000 times took 3.06635117531 seconds
# Result is [ 1.23178029  1.61879337  1.52278066 ...,  2.20771813  2.29967761
#   1.62323284]
# Used the cpu

# $ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python check1.py
# Using gpu device 0: GeForce GTX 580
# [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>), HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
# Looping 1000 times took 0.638810873032 seconds
# Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
#   1.62323296]
# Used the gpu
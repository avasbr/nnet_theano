from deepnet import MultilayerNet as mln
from deepnet.common import nnetact as na
from deepnet.common import nnetutils as nu
import numpy as np
from math import log

# this dataset isn't changing, so just hard-code these values
m = 1000
d = 40
k = 2


def load_london_dataset(base_path):
    ''' loads the london kaggle dataset '''
    X_tr = np.genfromtxt('%s/train.csv' % base_path, delimiter=',')
    y_tr = np.genfromtxt('%s/trainLabels.csv' % base_path, delimiter=',')
    X_te = np.genfromtxt('%s/test.csv' % base_path, delimiter=',')

    def encode_one_hot(y):
        y_one_hot = np.zeros((m, k))
        y_one_hot[range(m), y] = 1
        return y_one_hot
    y_oh = encode_one_hot(np.asarray(y_tr, dtype='int32'))

    return X_tr, y_oh, X_te


def write_pred_to_csv(y_pred):
    ''' writes the predictions to a file which can be submitted to kaggle '''
    pass

# load the dataset
print 'Loading the london dataset...'
base_path = '/home/bhargav/datasets/kaggle_data/london/dataset'
X, y, X_te = load_london_dataset(base_path)

X_tr, y_tr, X_val, y_val = nu.split_train_val(X, y, 0.6)

# neural network parameters
mln_1_params = {'d': d, 'k': k, 'num_hids': [648], 'activs': ['reLU', 'softmax'],
                'loss_terms': ['cross_entropy', 'regularization'], 'l2_decay': 1.842752802378564e-06, 'l1_decay': 0.0005302839353751183}

rmsprop_1_params = {'init_method': 'fan-io', 'scale_factor': 0.2686979643701496, 'optim_type': 'minibatch',
                    'optim_method': 'RMSPROP', 'batch_size': 600, 'num_epochs': 191, 'learn_rate': 0.0008333688149869977,
                    'rho': 0.47643610595875113, 'plotting': True}

mln_2_params = {'d': d, 'k': k, 'num_hids': [645], 'activs': ['reLU', 'softmax'],
                'loss_terms': ['cross_entropy', 'regularization'], 'l2_decay': 0, 'l1_decay': 0.0019508075031216578}

rmsprop_2_params = {'init_method': 'fan-io', 'scale_factor': 0.018237366158273965, 'optim_type': 'minibatch',
                    'optim_method': 'RMSPROP', 'batch_size': 600, 'num_epochs': 77, 'learn_rate': 0.0024691489372129713,
                    'rho': 0.8002533862612815, 'plotting': True}

'Fitting a neural network...'
nnet = mln.MultilayerNet(**mln_2_params)
nnet.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, **rmsprop_2_params)

print 'Performance on validation set:'
print 100 * nnet.score(X_val, y_val), '%'

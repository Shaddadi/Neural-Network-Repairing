
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, 'src')
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from src.reach import compute_unsafety
import tensorflow as tf
import multiprocessing
from utils import DATA
from copy import deepcopy
import scipy.io as sio
import torch
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16).double(),
            nn.ReLU(),
            nn.Linear(16, 16).double(),
            nn.ReLU(),
            nn.Linear(16, 16).double(),
            nn.ReLU(),
            nn.Linear(16, 2).double(),
        )

    def forward(self, x):
        y = self.model(x)
        return y

class CARTPOLE_V0:
    def __init__(self, filename):
        self.model_tensor = Sequential()
        self.model_tensor.add(Flatten(input_shape=(1,4)))
        self.model_tensor.add(Dense(16))
        self.model_tensor.add(Activation('relu'))
        self.model_tensor.add(Dense(16))
        self.model_tensor.add(Activation('relu'))
        self.model_tensor.add(Dense(16))
        self.model_tensor.add(Activation('relu'))
        self.model_tensor.add(Dense(2))
        self.model_tensor.add(Activation('linear'))
        print(self.model_tensor.summary())

        self.model_tensor.load_weights(modelname)

        self.convert_to_torch()


    def convert_to_torch(self):
        self.model_torch = Net().model
        layers = self.model_tensor.get_weights()
        k = 0
        for n in range(len(self.model_torch)):
            # assign to torch model
            if type(self.model_torch[n]).__name__ == 'Linear':
                weights = layers[k].T
                bias = layers[k+1].T
                with torch.no_grad():
                    self.model_torch[k].weight.data = torch.tensor(weights)
                    self.model_torch[k].bias.data = torch.tensor(bias)
                    k += 2
            else:
                continue


def test_conversion():
    modelname = '../examples/dqn_CartPole-v0_weights.h5f'
    model_torch = CARTPOLE_V0(modelname).model_torch
    xx = np.array([[0.020, -0.013, -0.0210, -0.0393]], dtype=np.float32)

    init_g = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        model_tensor = CARTPOLE_V0(modelname).model_tensor
        yy = sess.run(model_tensor(tf.convert_to_tensor(xx)))

    yy2 = model_torch(torch.tensor(xx))
    zz = 1


def main(candidate, datax):
    cpus = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(cpus)
    unsafe_data, safe_data, property_result, vfls_all, vfls_unsafe \
        = compute_unsafety(candidate, datax, pool)

    pool.close()

    sio.savemat('/logs/reach_sets.mat',
                {'all_reach_vfls': vfls_all, 'all_unsafe_vfls': vfls_unsafe})


if __name__ == '__main__':
    filename = 'dqn_CartPole-v0'
    modelname = '../examples/'+filename+'_weights.h5f'
    model_torch = CARTPOLE_V0(modelname).model_torch

    main(model_torch, DATA())



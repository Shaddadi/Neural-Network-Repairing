
from __future__ import print_function
import torch
import torch.nn as nn
import sys
sys.path.insert(0, 'src')
import torch.optim as optim
import torch.nn.functional as F
from src.reach import compute_unsafety
from src.correction import correct_inputs, get_safe_inputs
from torch.utils.data import TensorDataset, DataLoader
from utils import test, train
import tensorflow as tf
import multiprocessing
from utils import DATA
from copy import deepcopy
import scipy.io as sio
import torch
import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        y = self.model(x)
        return y

class CARTPOLE_V0:
    def __init__(self, modelname):

        self.model_torch = Net().model
        with open(modelname, 'rb') as f:
            layers = pickle.load(f)

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

        if torch.cuda.is_available():
            self.model_torch.cuda()


# Asymmetric loss function
def asymMSE(y_true, y_pred):
    lossFactor = 40.0
    numOut = 2
    d = y_true - y_pred
    maxes = torch.argmax(y_true, dim=1)
    maxes_onehot = torch.nn.functional.one_hot(maxes, numOut)
    others_onehot = maxes_onehot - 1
    d_opt = d * maxes_onehot
    d_sub = d * others_onehot
    a = lossFactor * (numOut - 1) * (torch.square(d_opt) + torch.abs(d_opt))
    b = torch.square(d_opt)
    c = lossFactor * (torch.square(d_sub) + torch.abs(d_sub))
    d = torch.square(d_sub)
    loss = torch.where(d_sub > 0, c, d) + torch.where(d_opt > 0, a, b)
    return torch.mean(loss)


def main(candidate, datax, epochs=300, batch_size=2**14, lr=0.001, steps=1):
    t0 = time.time()

    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('logs'):
        os.mkdir('logs')

    ori_train_x, ori_train_y = datax.train_x, datax.train_y
    optimizer = optim.Adam(candidate.parameters(), lr=lr)
    lossF = F.mse_loss
    cpus = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(cpus)

    all_test_accuracy = []
    all_reach_vfls = []
    all_unsafe_vfls = []
    all_property_result = []
    all_test_loss = []
    all_times = []

    # unsafe_data, safe_data, property_result, vfls_all, vfls_unsafe \
    #     = compute_unsafety(candidate, datax, pool)
    #
    # if len(safe_data[0]) != 0:
    #     ori_train_x = torch.cat((ori_train_x, safe_data[0]), dim=0)
    #     ori_train_y = torch.cat((ori_train_y, safe_data[1]), dim=0)
    #
    # all_property_result.append(property_result)
    # accuracy, mseloss = test(candidate.eval(), datax)
    # all_test_accuracy.append(accuracy)
    # all_test_loss.append(mseloss)

    for epoch in range(epochs):
        print('Epoch of training: ', epoch)

        unsafe_data, safe_data, property_result, vfls_all, vfls_unsafe\
            = compute_unsafety(candidate, datax, pool)

        plt.plot(unsafe_data[0][1][:,0].cpu(), unsafe_data[0][1][:,1].cpu(), 'r.')
        plt.plot(safe_data[0][1][:,0].cpu(), safe_data[0][1][:,1].cpu(), 'b.')
        plt.show()

        # all_safe_inputs = torch.cat([ls[0] for ls in all_safe_data])
        # all_safe_outputs = torch.cat([ls[1] for ls in all_safe_data])
        # if torch.cuda.is_available():
        #     all_safe_inputs = all_safe_inputs.cuda()
        #     all_safe_outputs = all_safe_outputs.cuda()

        safe_data = [torch.cat((safe_data[0][0],safe_data[1][0]),dim=0), torch.cat((safe_data[0][1],safe_data[1][1]),dim=0)]

        all_reach_vfls.append(vfls_all)
        all_unsafe_vfls.append(vfls_unsafe)
        all_property_result.append(property_result)
        accuracy, mseloss = test(candidate.eval(), datax)
        all_test_accuracy.append(accuracy)
        all_test_loss.append(mseloss)

        if np.all([len(aset[0])==0 for aset in unsafe_data]) and (accuracy>=0.94):
            print('\nThe accurate and safe candidate model is found !\n')
            test(candidate, datax)
            print('\n\n')
            torch.save(candidate.state_dict(), "models/acasxu_epoch" + str(epoch) + "_safe.pt")
            sio.savemat('logs/all_test_accuracy.mat', {'all_test_accuracy': all_test_accuracy, 'all_test_loss':all_test_loss})
            sio.savemat('logs/reach_sets.mat', {'all_reach_vfls': all_reach_vfls, 'all_unsafe_vfls':all_unsafe_vfls})
            sio.savemat('logs/all_property_result.mat', {'all_property_result': all_property_result})
            break


        if not np.all([len(aset[0])==0 for aset in unsafe_data]):
            unsafe_xs, corrected_ys = correct_inputs(unsafe_data,datax)
            print('  Unsafe_inputs: ', len(unsafe_xs))
            train_x = torch.cat((unsafe_xs, ori_train_x), dim=0)
            train_y = torch.cat((corrected_ys,ori_train_y), dim=0)
            if len(safe_data[0]) != 0:
                train_x = torch.cat((train_x, safe_data[0]), dim=0)
                train_y = torch.cat((train_y, safe_data[1]), dim=0)

        else:
            train_x = torch.cat((ori_train_x, safe_data[0]), dim=0)
            train_y = torch.cat((ori_train_y, safe_data[1]), dim=0)

        training_dataset = TensorDataset(train_x.cpu(), train_y.cpu())
        train_loader = DataLoader(training_dataset, batch_size, shuffle=False,num_workers=15)
        print('  Start training...')
        # train(candidate.train(), train_loader, lossF, optimizer)
        train(candidate.train(), train_loader, asymMSE, optimizer)
        print('  The training is done\n')

        if not np.all([len(aset[0])==0 for aset in unsafe_data]):
            safe_xs, safe_ys = get_safe_inputs(candidate, unsafe_data, datax)
            ori_train_x = torch.cat((ori_train_x, safe_xs), dim=0)
            ori_train_y = torch.cat((ori_train_y, safe_ys), dim=0)

        all_times.append(time.time()-t0)

        sio.savemat('logs/all_test_accuracy.mat', {'all_test_accuracy': all_test_accuracy})
        sio.savemat('logs/reach_sets.mat',{'all_reach_vfls': all_reach_vfls, 'all_unsafe_vfls': all_unsafe_vfls})
        sio.savemat('logs/all_property_result.mat', {'all_property_result': all_property_result})
        sio.savemat('logs/all_times.mat', {'all_times': all_times})
        if epoch % 1 == 0:
            torch.save(candidate.state_dict(), "models/acasxu_epoch"+str(epoch)+".pt")

    pool.close()



if __name__ == '__main__':
    filename = 'dqn_CartPole-v0'
    modelname = '../examples/'+filename+'_weights.pkl'
    model_torch = CARTPOLE_V0(modelname).model_torch.double()


    main(model_torch, DATA(model_torch))



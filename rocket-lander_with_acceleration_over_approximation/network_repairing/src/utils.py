
from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnetwork import *
import numpy as np
import copy as cp
from copy import deepcopy
import pickle
import math

def train(candidate, train_loader, lossFunction, optimizer):
    candidate.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        predicts = candidate(data.cuda())
        loss = lossFunction(target.cuda(), predicts)
        loss.backward()
        optimizer.step()


def test(model, datax):
    model.eval()
    predicts = model(datax.test_x)
    pred_actions = torch.argmax(predicts,dim=1)
    actl_actions = torch.argmax(datax.test_y,dim=1)
    accuracy = len(torch.nonzero(pred_actions==actl_actions))/len(predicts)
    mseloss = F.mse_loss(datax.test_y, predicts).detach().cpu().numpy()
    print('  Accuracy on the test data: {:.2f}% '.format(accuracy*100))
    print('  Mean loss on the test data: {:.5f} '.format(mseloss))
    return accuracy, mseloss



class DATA:
    def __init__(self,bs,normalizer=None,model=None,test_num=10000,samples_vol=100000):
        self.model = model
        self.test_num = test_num
        self.samples_vol = samples_vol
        self.normalizer = normalizer
        self.bs = deepcopy(bs)
        # self.bs[0][5] = -35 * math.pi / 180
        # self.bs[1][5] =  35 * math.pi / 180

        self.init_inputs()
        self.load_property()



    def purify_data(self, train_x, train_y):
        # maxx = torch.max(self.train_x, dim=0)
        # minn = torch.min(self.train_y, dim=0)
        for p in self.properties:
            lb, ub = p[0][0], p[0][1]
            M, vec = torch.tensor(p[1][0]), torch.tensor(p[1][1])
            bools = torch.ones(len(train_x), dtype=torch.bool)
            if torch.cuda.is_available():
                bools = bools.cuda()
                M = M.cuda()
                vec = vec.cuda()

            for n in range(len(lb)):
                lbx, ubx = lb[n], ub[n]
                x = train_x[:, n]
                bools = (x > lbx) & (x < ubx) & bools

            if len(torch.nonzero(bools)) == 0:
                break

            outs = torch.matmul(M, train_y.T) + vec
            unsafe_outs_bool = torch.all(outs<=0, dim=0) & bools
            safe_indx = torch.nonzero(~unsafe_outs_bool)[:,0]
            if torch.cuda.is_available():
                safe_indx = safe_indx.cuda()

            train_x = train_x[safe_indx]
            train_y = train_y[safe_indx]

        return train_x, train_y


    def init_inputs(self):

        self.lb_input = self.bs[0]
        self.ub_input = self.bs[1]

        # self.lb_p2 = deepcopy(self.bs[0])
        # self.lb_p2[0] = -0.1 # relative x distance
        # self.lb_p2[5] = 0.0 # angular velocity
        # self.ub_p2 = deepcopy(self.bs[1])
        # self.ub_p2[0] = 0.1  # relative x distance
        # self.ub_p2[1] = 0.1 # relative y distance
        # self.ub_p2[4] = -5*math.pi/180 # theta angle
        #
        # self.lb_p3 = deepcopy(self.bs[0])
        # self.lb_p3[0] = -0.1  # relative x distance
        # self.lb_p3[4] = 5*math.pi/180
        # self.ub_p3 = deepcopy(self.bs[1])
        # self.ub_p3[0] = 0.1  # relative x distance
        # self.ub_p3[1] = 0.1  # relative y distance
        # self.ub_p3[5] = 0.0

        self.lb_p2 = [-0.2, 0.02, -0.5, -1.0, -20*math.pi/180, -0.2, 0.0, 0.0, 0.0, -1.0, -15*math.pi/180]
        self.ub_p2 = [ 0.2, 0.5,  0.5,  1.0, -6*math.pi/180, -0.0, 0.0, 0.0, 1.0, 0.0, 0*math.pi/180]

        self.lb_p3 = [-0.2, 0.02, -0.5, -1.0, 6*math.pi/180, 0.0, 0.0, 0.0, 0.0, 0.0, 0*math.pi/180]
        self.ub_p3 = [ 0.2, 0.5,  0.5,  1.0, 20*math.pi/180,    0.2, 0.0, 0.0, 1.0, 1.0, 15*math.pi/180 ]

        if self.normalizer is not None:
            self.lb_p2 = self.normalizer(self.lb_p2)
            self.ub_p2 = self.normalizer(self.ub_p2)
            self.lb_p3 = self.normalizer(self.lb_p3)
            self.ub_p3 = self.normalizer(self.ub_p3)


    def load_property(self):
        # Fe = MainEngine(verticalthruster) [0, 1]
        # Fs = SideNitrogenThrusters[-1, 1]
        # Psi = Nozzleangle[-NOZZLE_LIMIT, NOZZLE_LIMIT]
        unsafe_mat_p2 = np.array([[0.0, -1.0, 0.0],[0.0, 0.0, -1.0]])
        unsafe_vec_p2 = np.array([[0.0],[0.0]])
        unsafe_mat_p3 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        unsafe_vec_p3 = np.array([[0.0],[0.0]])

        p2 = [[self.lb_p2, self.ub_p2], [unsafe_mat_p2, unsafe_vec_p2]]
        p3 = [[self.lb_p3, self.ub_p3], [unsafe_mat_p3, unsafe_vec_p3]]
        self.properties = [p2, p3]
        # self.properties = [p3]






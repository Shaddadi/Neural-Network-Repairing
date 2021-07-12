"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Train DDPG network in a different way.
"""


import sys
import os
import shutil
import argparse
import torch
import torch.nn as nn
from environments.rocketlander import get_state_sample
sys.path.insert(0, '../../network_repairing/src')
import tensorflow as tf
from reach import compute_unsafety
import multiprocessing
from utils import DATA
from constants import *
import numpy as np
import pickle
import scipy.io as sio
import random
import time
from control_and_ai.DDPG.ddpg import DDPG
from control_and_ai.DDPG.train import set_up
from control_and_ai.DDPG.train_third_model_unnormalized_reachability import train as train_third_model_unnormalized_reachability
from control_and_ai.DDPG.train_third_model_normalized_reachability import train as train_third_model_normalized_reachability
from constants import DEGTORAD
from control_and_ai.DDPG.exploration import OUPolicy
from environments.rocketlander import RocketLander


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 3),

        )

    def forward(self, x):
        y = self.model(x)
        return y


class Rocket_Lander:
    def __init__(self, layers):

        self.model_torch = Net().model

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


def setup_reachability(agent, filename):
    weights = agent.sess.run(agent.actor.weights)
    model_torch = Rocket_Lander(weights).model_torch.double()

    datax = DATA([[], []])

    # load old buffer for agent
    with open(filename, 'rb') as f:
        buffer_old = pickle.load(f)
        length = len(buffer_old) // 1
        buffer_old = purify_buffer(buffer_old[-length:], datax)  # remove unsafe training data

    agent.load_old_buffer(buffer_old=buffer_old)

    return model_torch, datax


def purify_buffer(buffer, datax):# remove unsafe training data
    purified_experiences = []
    for experience in buffer:
        states = np.array(experience[0])
        actions = np.array([experience[1]])
        unsafe_flag = False
        for pty in datax.properties:
            lb, ub = np.array(pty[0][0]), np.array(pty[0][0])
            A, d = pty[1][0], pty[1][1]
            bools_in =  np.all((states - lb) >= 0) and np.all((ub - states) >= 0)
            bools_out = np.all((np.dot(A,actions.T) + d)<=0)
            if bools_in and bools_out:
                unsafe_flag = True
                break

        if not unsafe_flag:
            purified_experiences.append(experience)

    return purified_experiences


if __name__ == "__main__":
    FLAGS = set_up()

    action_bounds = [1, 1, 15*DEGTORAD]

    eps = []
    eps.append(OUPolicy(0, 0.2, 0.4))
    eps.append(OUPolicy(0, 0.2, 0.4))
    eps.append(OUPolicy(0, 0.2, 0.4))

    simulation_settings = {'Side Engines': True,
                           'Clouds': True,
                           'Vectorized Nozzle': True,
                           'Graph': False,
                           'Render': False,
                           'Starting Y-Pos Constant': 1,
                           'Initial Force': 'random',
                           'Rows': 1,
                           'Columns': 2,
                           'Episodes': 300}
    env = RocketLander(simulation_settings)

    FLAGS.retrain = False # Restore weights if False
    FLAGS.test = False
    FLAGS.num_episodes = 500
    model_dir = './models_unnorm0'

    agent = DDPG(
        action_bounds,
        eps,
        env.observation_space.shape[0],
        actor_learning_rate=0.0001,
        critic_learning_rate=0.001,
        batch_size=1000,
        reachability=True,
        retrain=FLAGS.retrain,
        log_dir=FLAGS.log_dir,
        model_dir=model_dir)

    old_buffer = 'models_unnorm/buffer850_old.pkl'
    model_torch, datax = setup_reachability(agent,old_buffer)

    t0 = time.time()
    unsafe_data, property_result = compute_unsafety(model_torch, datax, over_app=True)
    print(time.time()-t0)

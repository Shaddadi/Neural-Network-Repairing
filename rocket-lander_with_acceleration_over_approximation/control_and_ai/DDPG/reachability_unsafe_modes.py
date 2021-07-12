"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Train DDPG network in a different way.
"""
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../../network_repairing/src')
import torch
import torch.nn as nn
from control_and_ai.DDPG.ddpg import DDPG
from control_and_ai.DDPG.train import set_up
from control_and_ai.DDPG.train_third_model_unnormalized_reachability import train as train_third_model_unnormalized_reachability
from control_and_ai.DDPG.train_third_model_normalized_reachability import train as train_third_model_normalized_reachability
from constants import DEGTORAD
from control_and_ai.DDPG.exploration import OUPolicy
from environments.rocketlander import RocketLander
import pickle
import tensorflow as tf
import os
from utils import DATA
from reach_hscc import compute_unsafety_hscc
from reach import compute_unsafety
import scipy.io as sio
import psutil


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(11, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 3),

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


FLAGS = set_up()

action_bounds = [0.5, 1, 15*DEGTORAD]

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
model_dirs = ['./models_unnorm1000(2)', './models_unnorm950(2)','./models_unnorm1000(1)'] #['./models_unnorm950', './models_unnorm1000']
train_idss =[[1,2,3,4,5],[11,12,13,14,15],[1,2,3,4,5]]
# model_dirs = ['./models_unnorm1000(1)'] #['./models_unnorm950', './models_unnorm1000']
# train_idss =[[3,4,5]]


# print('train_times:', train_times)
# for train_times in range(6,7):

def choose_model(num, filenames, target_dir):
    string_part1 = ''
    for ii in range(len(filenames[0])):
        string_part1 += filenames[0][ii]
        if filenames[0][ii] == '-':
            break

    filenames[0] = string_part1 + str(num) + '"\n'
    filenames = ''.join(filenames)
    with open(target_dir + '/checkpoint', 'w') as file:
        file.write(filenames)


for ii, model_dir in enumerate(model_dirs):
    train_ids = train_idss[ii]
    for train_id in train_ids:
        target_dir = model_dir+'/'+str(train_id)
        with open(target_dir + '/checkpoint', 'r') as file:
            filenames = file.readlines()

        models_num = len(filenames)-1
        all_model_time = []
        all_memory_usage = []
        for num in range(models_num):
            choose_model(num, filenames, target_dir)

            agent = DDPG(
                action_bounds,
                eps,
                env.observation_space.shape[0],
                actor_learning_rate=0.00001,
                critic_learning_rate=0.0001,
                batch_size=1000,
                batch_size_unsafe=50,
                buffer_size=200,
                reachability=True,
                retrain=FLAGS.retrain,
                log_dir=FLAGS.log_dir,
                model_dir=target_dir)

            weights = agent.sess.run(agent.actor.weights)
            model_torch = Rocket_Lander(weights).model_torch.double()
            datax = DATA([[], []])

            # _, running_time, used_memory1 = compute_unsafety_hscc(model_torch, datax)
            _, running_time, used_memory1 = compute_unsafety(model_torch, datax, over_app=True)

            print(target_dir)
            print('models_num ',num)
            print('running_time without relaxation: ', running_time)
            print('memory usage: ', used_memory1)
            all_model_time.append(running_time)
            all_memory_usage.append(used_memory1)

            tf.reset_default_graph()

        sio.savemat(target_dir + 'hscc.mat', {'all_model_time': all_model_time, 'all_memory_usage': all_memory_usage})



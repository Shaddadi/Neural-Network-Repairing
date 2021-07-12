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
from accuracy_evaluate import start_test
from accuracy_evaluate import test
from utils import DATA
from constants import *
from .utils import Utils
import numpy as np
import pickle
import scipy.io as sio
import random
import time
import copy as cp
from environments.rocketlander import RocketLander

simulation_settings = {'Side Engines': True,
                       'Clouds': True,
                       'Vectorized Nozzle': True,
                       'Graph': False,
                       'Render': False,
                       'Starting Y-Pos Constant': 1,
                       'Initial Force': 'random',
                       'Rows': 1,
                       'Columns': 2,
                       'Episodes': 50}

env_test = RocketLander(simulation_settings)

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


def setup_reachability(agent, filename):
    weights = agent.sess.run(agent.actor.weights)
    model_torch = Rocket_Lander(weights).model_torch.double()

    datax = DATA([[], []])

    # load old buffer for agent
    with open(filename, 'rb') as f:
        buffer_old = pickle.load(f)
        length = len(buffer_old)
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


def compute_unsafe_data(agent, datax):
    weights = agent.sess.run(agent.actor.weights)
    model_torch = Rocket_Lander(weights).model_torch.double()
    # unsafe_data, running_time = compute_unsafety(model_torch, datax, over_app=False)
    # print('Unsafety computation without Over Appro: ', running_time)
    unsafe_data2, running_time2 = compute_unsafety(model_torch, datax, over_app=True)
    print('Unsafety computation with Over Appro: ', running_time2)

    return unsafe_data2, [running_time2, running_time2]


def check_process_unsafe_data(unsafe_data, agent, base_reward):
    accurate_model = False
    safe_model = False
    for count, sublist in enumerate(unsafe_data):
        if len(sublist) > 1000:
            unsafe_data[count] = random.sample(sublist, 1000)
    unsafe_inputs = [[item] for sublist in unsafe_data for item in sublist]
    random.shuffle(unsafe_inputs)
    if len(unsafe_inputs) == 0:
        safe_model = True
        averaged_reward, done = start_test(agent, base_reward, env_test, simulation_settings)
        if done:
            accurate_model = True
            return safe_model, accurate_model, None, averaged_reward
        else:
            print('Safe but not accurate agent is found!')
            return safe_model, accurate_model, None, averaged_reward
    else:
        unsafe_inputs = np.concatenate(unsafe_inputs, axis=0)
        print('num_unsafe_inputs: ', len(unsafe_inputs))
        return safe_model, accurate_model, unsafe_inputs, None


def train_with_unsafe_states(env, agent, obs_size, FLAGS, unsafe_inputs):
    # unsafe_inputs = [[ 0.2 , 0.02 , -0.5, 1. , -0.12125815, -0.00617011,  0.,    0.  ,        0.38216249,  0.  , -0.26179939]]
    for episode, init_state in enumerate(unsafe_inputs):
        old_state = None
        done = False
        total_reward = 0

        s = env.reset(init_state=init_state)
        state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
        state = state[:11]
        max_steps = 5

        left_or_right_barge_movement = np.random.randint(0, 2)
        epsilon = 0.05

        for t in range(max_steps):  # env.spec.max_episode_steps
            if FLAGS.show or episode % 50 == 0:
                env.refresh(render=True)

            old_state = state

            # infer an action
            action = agent.get_action(np.reshape(state, (-1, obs_size)), explore=False)

            # take it
            s, reward, done, _ = env.step(action[0], unsafe_training=True)
            state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
            state = state[:11]
            total_reward += reward

            if s[LEFT_GROUND_CONTACT] == 0 and s[RIGHT_GROUND_CONTACT] == 0:
                env.move_barge_randomly(epsilon, left_or_right_barge_movement)
                env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
                env.apply_random_y_disturbance(epsilon=0.005)

            if not FLAGS.test:
                # update q vals
                agent.update(old_state[:obs_size], action[0], np.array(reward), state[:obs_size], done)

            if done:
                break

        agent.log_data(total_reward, episode)

        print("Reward:\t{0}".format(total_reward))


def train_for_accuracy(agent, base_reward, best_reward, datax, train_id, iteration, running_times_computation_unsafety):
    num = 0
    ii = 0
    best_agent = cp.copy(agent)
    while num<10:
        for episode in range(1000):
            agent.update([], [], [], [], [], with_new_state=False)

        averaged_reward, done = start_test(agent, base_reward, env_test, simulation_settings)
        print('train_for_accuracy, averaged_reward: ', averaged_reward)
        if averaged_reward>best_reward:
            unsafe_data, rtime = compute_unsafe_data(agent, datax)
            running_times_computation_unsafety.append(rtime)
            agent.model_loc = agent.model_dir + '/' + str(train_id) + '/DDPG_unsafe.ckpt'
            agent.save_model(iteration+ii)
            if np.all([not p for p in unsafe_data]):
                best_reward = averaged_reward
                best_agent = cp.copy(agent)
            else:
                agent = cp.copy(best_agent)
                done = False
                print('Not safe')

            ii += 1

        num = num+1
        if done:
            break

    return best_agent, best_reward


def train(env, agent, FLAGS, train_id=0):
    print("Fuel Cost = 0.3, Max Steps = Unlimited, Episode Training = 1000, RANDOM FORCE = 20000, RANDOM X_FORCE = 0.2*RANDOM FORCE")
    print("4th Model with get_state_with_barge_and_landing_coordinates - Normalized")
    print("Using Normal State nor Untransformed. Otherwise same settings as previous unnormalized test.")
    obs_size = env.observation_space.shape[0]

    old_buffer = 'models_unnorm/buffer800_old.pkl'
    model_torch, datax = setup_reachability(agent,old_buffer)
    base_reward = test(env, agent, simulation_settings)
    print('Base reward: ', base_reward)

    learning_time0 = time.time()
    running_times_computation_unsafety = []
    iteration = 0
    while True:
        print('Iteration: ', iteration)
        agent.model_loc = agent.model_dir + '/'+str(train_id)+'/DDPG_unsafe.ckpt'
        agent.save_model(iteration)

        unsafe_data, running_times = compute_unsafe_data(agent, datax)
        running_times_computation_unsafety.append(running_times)

        safe_model, accurate_model, unsafe_inputs, averaged_reward = check_process_unsafe_data(unsafe_data, agent, base_reward)
        iteration += 1
        # if safe_model and accurate_model:
        #     break

        # if safe_model:
        #     break

        if not safe_model:
            train_with_unsafe_states(env, agent, obs_size, FLAGS, unsafe_inputs)
        elif safe_model and not accurate_model:
            agent, averaged_reward = train_for_accuracy(agent, base_reward, averaged_reward, datax, train_id, iteration,running_times_computation_unsafety)
            break
        else:
            break

    print('Safe and accurate agent is found!')
    sio.savemat('models_unnorm/results' + str(train_id) + '.mat',
                {'all_learning_time': time.time() - learning_time0,
                 'all_computation_unsafety': running_times_computation_unsafety,
                 'iterations': iteration,
                 'averaged_reward': averaged_reward})

    agent.model_loc = agent.model_dir + '/models_safe/DDPG_safe.ckpt'
    agent.save_model(train_id)



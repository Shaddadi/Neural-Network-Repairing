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
from .utils import Utils
import numpy as np
import pickle


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

def setup_reachability(agent):
    weights = agent.sess.run(agent.actor.weights)
    model_torch = Rocket_Lander(weights).model_torch.double()
    with open('state_samples.pkl', 'rb') as f:
        state_samples = pickle.load(f)

    util = Utils()
    # state_samples = get_state_sample(samples=5000, normal_state=False, untransformed_state=False)
    with open('state_samples.pkl', 'rb') as f:
        state_samples = pickle.load(f)
        state_samples = [ls[:8] for ls in state_samples]
        state_samples = np.array(state_samples)
        bound_lower_samples = np.min(state_samples, axis=0)
        bound_upper_samples = np.max(state_samples, axis=0)

    with open('all_reached_states_unnorm.pkl', 'rb') as f:
        all_reached_states = pickle.load(f)
        all_reached_states = [ls[:8] for ls in all_reached_states]
        all_reached_states = np.array(all_reached_states[-10000:])
        bound_lower_reached = np.min(all_reached_states, axis=0)
        bound_upper_reached = np.max(all_reached_states, axis=0)

    util.create_normalizer(state_sample=state_samples)
    normalizer = util.normalize
    datax = DATA([bound_lower_reached, bound_upper_reached], normalizer)

    return model_torch, datax


def train(env, agent, FLAGS):
    print("Fuel Cost = 0.3, Max Steps = Unlimited, Episode Training = 1000, RANDOM FORCE = 20000, RANDOM X_FORCE = 0.2*RANDOM FORCE")
    print("4th Model with get_state_with_barge_and_landing_coordinates - Normalized")
    print("Using Normal State nor Untransformed. Otherwise same settings as previous unnormalized test.")
    #print("Fuel Cost = 0, Max Steps = Unlimited, Episode Training = 2000")
    obs_size = env.observation_space.shape[0]

    model_torch, datax = setup_reachability(agent)
    cpus = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(cpus)



    for episode in range(1, FLAGS.num_episodes + 1):

        unsafe_data, safe_data, property_result, vfls_all, vfls_unsafe \
            = compute_unsafety(model_torch, datax, pool)

        old_state = None
        done = False
        total_reward = 0

        s = env.reset()
        state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
        state = util.normalize(np.array(state))
        max_steps = 1000

        left_or_right_barge_movement = np.random.randint(0, 2)
        epsilon = 0.05

        for t in range(max_steps): # env.spec.max_episode_steps
            if FLAGS.show or episode % 10 == 0:
                env.refresh(render=True)

            old_state = state

            # infer an action
            action = agent.get_action(np.reshape(state, (-1, obs_size)), not FLAGS.test)

            # take it
            s, reward, done, _ = env.step(action[0])
            state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
            state = util.normalize(np.array(state))
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

        if episode % 50 == 0 and not FLAGS.test:
            print('Saved model at episode', episode)
            agent.save_model(episode)
        print("Reward:\t{0}".format(total_reward))
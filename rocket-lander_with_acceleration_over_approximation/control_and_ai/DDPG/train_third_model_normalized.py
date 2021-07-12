"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Train DDPG network in a different way.
"""

import sys
import os
import shutil
import argparse
#sys.path.append('C://Users//REUBS_LEN//PycharmProjects//RocketLanding')
from environments.rocketlander import get_state_sample
from constants import *
from .utils import Utils
import numpy as np
import pickle


def train(env, agent, FLAGS):
    print("Fuel Cost = 0.3, Max Steps = Unlimited, Episode Training = 1000, RANDOM FORCE = 20000, RANDOM X_FORCE = 0.2*RANDOM FORCE")
    print("4th Model with get_state_with_barge_and_landing_coordinates - Normalized")
    print("Using Normal State nor Untransformed. Otherwise same settings as previous unnormalized test.")
    #print("Fuel Cost = 0, Max Steps = Unlimited, Episode Training = 2000")
    obs_size = env.observation_space.shape[0]

    util = Utils()
    # state_samples = get_state_sample(samples=5000, normal_state=False, untransformed_state=False)
    # with open('state_samples.pkl', 'wb') as f:
    #     pickle.dump(state_samples, f)
    # with open('state_samples.pkl', 'rb') as f:
    #     state_samples = pickle.load(f)
    # util.create_normalizer(state_sample=state_samples)

    var = [1.0, 2.0, 3.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var = np.array(var)

    all_reached_states_unnorm = []
    for episode in range(1, FLAGS.num_episodes + 1):
        old_state = None
        done = False
        total_reward = 0

        s = env.reset()
        state = env.get_state_with_barge_and_landing_coordinates(untransformed_state=False)
        all_reached_states_unnorm.append(state)
        # state = util.normalize(np.array(state))
        state = np.divide(np.array(state),var)
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
            all_reached_states_unnorm.append(state)
            # state = util.normalize(np.array(state))
            state = np.divide(np.array(state), var)
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


    with open('all_reached_states_unnorm.pkl', 'wb') as f:
        pickle.dump(all_reached_states_unnorm, f)
"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Not a unit test, simple run test.
"""
from control_and_ai.DDPG.train import set_up
import numpy as np
from control_and_ai.DDPG.train import Utils
from environments.rocketlander import get_state_sample
from control_and_ai.DDPG.ddpg import DDPG
from constants import DEGTORAD
from control_and_ai.DDPG.exploration import OUPolicy
from environments.rocketlander import RocketLander
import pickle
import os

settings = {'Side Engines': True,
            'Clouds': True,
            'Vectorized Nozzle': True,
            'Starting Y-Pos Constant': 1,
            'Initial Force': 'random'}  # (6000, -10000)}

action_bounds = [1, 1, 15 * DEGTORAD]
eps = []
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))


left_or_right_barge_movement = np.random.randint(0, 2)
epsilon = 0.05


def test(env, agent, simulation_settings):
    obs_size = env.observation_space.shape[0]

    all_total_rewards = []
    for episode in range(1, simulation_settings['Episodes']):

        done = False
        total_reward = 0
        state = env.reset()

        for i in range(1000): #
            env.move_barge_randomly(epsilon, left_or_right_barge_movement)
            # Random Force on rocket to simulate wind.
            env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
            env.apply_random_y_disturbance(epsilon=0.005)

            if simulation_settings['Render']:
                env.refresh(render=True)
                # env.render()

            # infer an action
            action = agent.get_action(np.reshape(state, (1, obs_size)), explore=False)
            old_state = list(state)
            # take it
            state, reward, done, _ = env.step(action[0])
            new_state = list(state)
            agent.experience_buffer.add([old_state, action[0,:], np.array(reward), new_state, done])
            total_reward += reward

            if done:
                break

        agent.log_data(total_reward, episode)
        print("Reward:\t{0}".format(total_reward))
        all_total_rewards.append(total_reward)

    with open('models_unnorm/buffer' + '_old.pkl', 'wb') as f:
        pickle.dump(agent.experience_buffer.experiences, f)

    print('Averaged Reward: ', np.mean(all_total_rewards))
    return np.mean(all_total_rewards)


def start_test(agent, base_reward, env, simulation_settings):
    averaged_reward = test(env, agent, simulation_settings)
    done = (averaged_reward-base_reward)>=0
    return averaged_reward, done

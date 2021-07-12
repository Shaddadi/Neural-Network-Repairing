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


def test(env, agent, simulation_settings):
    obs_size = env.observation_space.shape[0]
    util = Utils()
    #state_samples = get_state_sample(samples=5000, normal_state=True)
    # with open('state_samples.pkl', 'rb') as f:
    #     state_samples = pickle.load(f)
    # state_samples = [ls[:8] for ls in state_samples]
    # util.create_normalizer(state_sample=state_samples)

    # var = [1.0, 2.0, 3.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # var = np.array(var[:8])
    all_total_rewards = []
    for episode in range(1, simulation_settings['Episodes']):

        done = False
        total_reward = 0
        state = env.reset()
        # state = util.normalize(state)
        # state = np.divide(np.array(state), var)

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

            # take it
            state, reward, done, _ = env.step(action[0])
            # state = util.normalize(state)
            # state = np.divide(np.array(state), var)
            total_reward += reward

            if done:
                break

        agent.log_data(total_reward, episode)
        print("Reward:\t{0}".format(total_reward))
        all_total_rewards.append(total_reward)
    print('Averaged Reward: ', np.mean(all_total_rewards))
    with open('models_unnorm/base_reward.pkl', 'wb') as f:
        pickle.dump(np.mean(all_total_rewards), f)



settings = {'Side Engines': True,
            'Clouds': True,
            'Vectorized Nozzle': True,
            'Starting Y-Pos Constant': 1,
            'Initial Force': 'random'}  # (6000, -10000)}

action_bounds = [1, 1, 15*DEGTORAD]
eps = []
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))
eps.append(OUPolicy(0, 0.2, 0.4))

simulation_settings = {'Side Engines': True,
                       'Clouds': True,
                       'Vectorized Nozzle': True,
                       'Graph': False,
                       'Render': True,
                       'Starting Y-Pos Constant': 1,
                       'Initial Force': 'random',
                       'Rows': 1,
                       'Columns': 2,
                       'Episodes': 50}

env = RocketLander(simulation_settings)
left_or_right_barge_movement = np.random.randint(0, 2)
epsilon = 0.05

test_safe_model = False
if test_safe_model:
    filename = os.getcwd() + '/models_unnorm/models_safe'
else:
    filename = os.getcwd() + '/models_unnorm'

agent = DDPG(
    action_bounds,
    eps,
    env.observation_space.shape[0],
    actor_learning_rate=0.0001,
    critic_learning_rate=0.001,
    retrain=False,
    log_dir='.',
    model_dir=filename)



test(env, agent, simulation_settings)

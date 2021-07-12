"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Train DDPG network in a different way.
"""
import os.path

from control_and_ai.DDPG.ddpg import DDPG
from control_and_ai.DDPG.train import set_up
from control_and_ai.DDPG.train_third_model_normalized import train as train_third_model_normalized
from control_and_ai.DDPG.train_third_model_unnormalized import train as train_third_model_unnormalized
from constants import DEGTORAD
from control_and_ai.DDPG.exploration import OUPolicy
from environments.rocketlander import RocketLander
import pickle

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
#env = wrappers.Monitor(env, '/tmp/contlunarlander', force=True, write_upon_reset=True)

FLAGS.retrain = True # Restore weights if False
FLAGS.test = False
FLAGS.num_episodes = 1000
model_dir = './models_unnorm'
if not os.path.isdir(model_dir):
    os.mkdir('model_dir')

agent = DDPG(
    action_bounds,
    eps,
    env.observation_space.shape[0],
    buffer_size=400000,
    actor_learning_rate=0.0001,
    critic_learning_rate=0.001,
    batch_size = 400,
    retrain=FLAGS.retrain,
    log_dir=FLAGS.log_dir,
    model_dir=model_dir)

#test(env, agent, simulation_settings)
# train_third_model_normalized(env, agent, FLAGS)
train_third_model_unnormalized(env, agent, FLAGS)

#train_second_model(env, agent, FLAGS)


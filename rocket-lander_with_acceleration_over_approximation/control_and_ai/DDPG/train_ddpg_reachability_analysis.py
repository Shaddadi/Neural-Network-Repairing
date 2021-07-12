"""
Author: Reuben Ferrante
Date:   10/05/2017
Description: Train DDPG network in a different way.
"""
import sys
sys.path.insert(0, '../../')
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
model_dir = './models_unnorm'
train_id = 5
if not os.path.isdir(model_dir+'/'+str(train_id)):
    os.mkdir(model_dir+'/'+str(train_id))

# train_times = int(sys.argv[1])

# print('train_times:', train_times)
# for train_times in range(6,7):
agent = DDPG(
    action_bounds,
    eps,
    env.observation_space.shape[0],
    actor_learning_rate=0.00001,
    critic_learning_rate=0.0001,
    batch_size=1000,
    batch_size_unsafe=20,
    buffer_size=200,
    reachability=True,
    retrain=FLAGS.retrain,
    log_dir=FLAGS.log_dir,
    model_dir=model_dir)

train_third_model_unnormalized_reachability(env, agent, FLAGS, train_id=train_id)

tf.reset_default_graph()

#train_second_model(env, agent, FLAGS)


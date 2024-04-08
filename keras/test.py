import numpy as np
import gym
import tensorflow.keras.optimizers as opt
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, BoltzmannQPolicy
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import tunnel
import time
from keras.regularized_agent import RegularizedAgent, egreedy, boltzmann
from keras.layers import *
from keras.model import make_model
from keras.callbacks import *
import tensorflow.keras.backend as K


np.set_printoptions(linewidth=2000, precision=3)

def run_test(env, agent, episodes, regul_factor, render, reset_tunnel=True, log_name=''):
    seed = str(time.time())
    cbk_list = [HistoryAccumulator(env),
                RewardsCallback(env),
                LogWriterCallback(log_dir=f'./logs/test-{log_name}')] # f'./logs/test-{seed}-{log_name}'

    boltzmann_tau = 0.1
    agent.num_masks = 10
    env.unwrapped.reset_tunnel = reset_tunnel
    agent.test(episodes, regul_factor=regul_factor, mask_regul=0.05,
               boltzmann_tau=boltzmann_tau, render=render, callbacks=cbk_list)
    env.unwrapped.reset_tunnel = True
    cbk = cbk_list[0]
    return cbk



def test_and_exit(env, agent, params_path):
    agent.load_weights(params_path)
    ag = RegularizedAgent(agent, env, boltzmann_tau=0.2)


    env.unwrapped.steer_reward = 0.
    env.unwrapped.clear_on_add = False
    env.unwrapped.base_reward_color = 0
    env.unwrapped.color_density = 14
    env.unwrapped.obstacle_density = 4
    env.unwrapped.num_colors = 3
    env.unwrapped.goal_size = 4
    env.unwrapped.reset_on_obstacle_hit = False
    n_tests = 30
    render = False
    pos_list = [
                ((8, 12), (4, 7)),
                ((4, 19), (4, 7)),
                ((3, 19), (7, 19)),

                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),

                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ((3, 19), (7, 19)),
                ]

    for i, positions in enumerate(pos_list):
        pos1, pos2 = positions
        env.unwrapped.pos1 = pos1
        env.unwrapped.pos2 = pos2
        env.unwrapped.obstacles = [(1, 14), (1, 1),]
        env.reset()
        for f in [0, 0.1, 0.5, 2, 5]:
            print(f'Regularization factor: {f}')
            cbk = run_test(env, ag, n_tests, regul_factor=f, render=render, reset_tunnel=False, log_name=f'{i}-{f}')
            #cbk.save_plot(f'figure-{i}-{f}')

    exit(0)




if __name__ =='__main__':

    ENV_NAME = 'Tunnel-v0'

    #ENV_NAME = 'TestTunnel-v0'

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)

    env = tunnel.SlicedEmbeddings(env)


    nb_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    model = make_model(env, nb_actions, mask_regul=1e-6, mask_regul_temp=0.1, n_masks=None, masks_dropout=0.3)

    #from keras.model_0 import get_model_0
    #model = get_model_0(env)

    model.build(input_shape=(None,)+env.observation_space.shape)

    memory_size = 150000
    warmup = 50000
    memory = SequentialMemory(limit=memory_size, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', value_max=.2, value_min=.2, value_test=.15, nb_steps=200000)
    agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy,
                     batch_size=100, nb_steps_warmup=warmup,
                     train_interval=5, gamma=0.98, enable_double_dqn=True, target_model_update=1e-2)

    optimizer = opt.Adam(1e-3, amsgrad=True)


    with tf.keras.utils.custom_object_scope({'TokenAndPositionEmbedding' : TokenAndPositionEmbedding}): # Custom scope to load token layer class
        agent.compile(optimizer, metrics=['mae',])

    ''' Test a pre-trained agent '''
    test_and_exit(env, agent, './logs/model_x.bak/model_params.h5f')

    ''' Load a pre-trained agent '''
    agent.load_weights('./logs/model_x/model_params.h5f')


    log_dir = f'./logs/{time.time()}'
    callbacks = [
        WriterCallback(agent.model, log_dir=log_dir),
        Scheduler(callback=lambda x: K.set_value(optimizer.lr, x), episodes=4e4, min_value=1e-3, max_value=1e-5, log_name='lr'),
        RewardsCallback(env),
        LogWriterCallback(log_dir=log_dir),
    ]



    agent.fit(env, nb_steps=50000000, visualize=False, verbose=2,
              callbacks=callbacks)


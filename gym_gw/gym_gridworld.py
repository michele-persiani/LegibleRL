import gym
import highway_env
import gym_minigrid
from gym_minigrid.wrappers import *
from collections import deque
import numpy as np
from rl_agents.agents.deep_q_network.pytorch import DQNAgent
from rl_agents.trainer.evaluation import Evaluation
import tunnel
import torch

"""
'model' : {
        'type' : 'ConvolutionalNetwork',
        "layers" : [{
            "in_channels" : 6,
            "out_channels" : 50,
            "kernel_size" : (1,8),
            "stride" : 1,
        },{
            "in_channels" : 50,
            "out_channels" : 50,
            "kernel_size" : (1,8),
            "stride" : 1,
        },{
            "in_channels" : 50,
            "out_channels" : 50,
            "kernel_size" : (1,6),
            "stride" : 1,
        },],
        "head_mlp": {
            "type": "MultiLayerPerceptron",
            "in": None,
            "layers": [50, 50],
            "activation": "RELU",
            "reshape": "True",
            "out": None
        }
    },
'model': {
    'type': 'MultiLayerPerceptron',
    "layers": [200, 200, 200],
    'activation': 'RELU',
},
"""

config = {
'model' : {
        'type' : 'ConvolutionalNetwork',
        "layers" : [{
            "in_channels" : 9,
            "out_channels" : 40,
            "kernel_size" : (1,8),
            "stride" : 1,
        },{
            "in_channels" : 40,
            "out_channels" : 40,
            "kernel_size" : (1,8),
            "stride" : 1,
        },{
            "in_channels" : 40,
            "out_channels" : 40,
            "kernel_size" : (1,6),
            "stride" : 1,
        },],
        "head_mlp": {
            "type": "MultiLayerPerceptron",
            "in": None,
            "layers": [],
            "activation": "RELU",
            "reshape": "True",
            "out": None
        }
    },
    'device' : 'cuda',
    'double' : True,
    'target_update' : 10,
    'optimizer' : {
        'lr' : 2e-4
    },
    'gamma' : 0.99,
    'batch_size' : 100,
    "memory_capacity": 50000,
    "exploration": {
        "method": "EpsilonGreedy",
        "tau": 20000,
        "temperature": 1.0,
        "final_temperature": 0.2
    }
}


import gc
def episode_cbk(episode, logs):
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':




    #env = gym.make('highway-v0')
    #env = gym.make('MiniGrid-DoorKey-6x6-v0')
    #env = FlatObsWrapper(env)
    #env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    #env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    env = gym.make('Tunnel-v0')
    state = env.reset()
    agent = DQNAgent(env, config=config)

    trainer = tunnel.EnvTraining(env, agent, logs_directory='./logs', render=False)
    trainer.train(num_episodes=100000, callbacks=[episode_cbk, ])
    exit(0)


    ev = Evaluation(env, agent,
                    directory='./logs',
                    run_directory='./',
                    num_episodes=10,
                    training=True,
                    sim_seed=0,
                    recover=None,
                    display_env=True,
                    display_agent=True,
                    display_rewards=True,
                    close_env=False)

    ev.train()
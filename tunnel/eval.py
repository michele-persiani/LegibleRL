from collections import deque, defaultdict
from tensorboardX import SummaryWriter
import numpy as np
import shutil
import os
import time



class EnvTraining:



    def __init__(self, env, agent, logs_directory='./logs', render=True, max_num_steps=1e16):
        self.env = env
        self.agent = agent

        seed = str(time.time())
        self.logs_directory = f'{logs_directory}/{seed}'
        os.mkdir(self.logs_directory)
        self.writer = SummaryWriter(str(self.logs_directory))
        self.agent.set_writer(self.writer)
        self.max_num_steps = int(max_num_steps)
        self._render = render
        self.logs = {}
        self.global_logs = defaultdict(lambda : deque(maxlen=200))

    def log_avg(self, key):
        return np.mean(self.global_logs[key])

    def render(self):
        if self._render:
            self.env.render()


    def train(self, num_episodes=1e16, callbacks=None):
        env = self.env
        agent = self.agent
        logs = self.global_logs

        if callbacks is None:
            callbacks = []

        for e in range(int(num_episodes)):
            state = env.reset()
            self.render()
            for step in range(self.max_num_steps):
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.record(state, action, reward, next_state, done, info)
                self.render()

                if done:
                    logs['accumulated_reward'] += info['accumulated_reward'],
                    logs['max_accumulated_reward'] += info['max_accumulated_reward'],
                    logs['epsilon'] += agent.exploration_policy.epsilon,
                    logs['steps'] += step + 1,
                    print(f'Episode: {e}  Reward: {int(self.log_avg("accumulated_reward"))}/{int(self.log_avg("max_accumulated_reward"))}')
                    break
                state = next_state

            for cbk in callbacks:
                cbk(e, logs)

            self.on_episode_end(e)


    def on_episode_end(self, episode):
        self.print_logs(episode)
        self.agent.save('./models/agent')

    def print_logs(self, episode, prefix=None):
        logs = self.global_logs
        for k in logs.keys():
            v = self.log_avg(k)
            if prefix is not None:
                k = f'{prefix}/{k}'
            self.writer.add_scalar(k, v, episode)
        self.writer.flush()
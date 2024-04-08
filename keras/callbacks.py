import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorboardX import SummaryWriter
from tensorflow.python.keras import backend
import rl.callbacks as cbks
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from collections import defaultdict
import time
from PIL import Image as PImage

def sample_batch(agent, batch_size):
    experiences = agent.memory.sample(batch_size)
    assert len(experiences) == batch_size

    # Start by extracting the necessary parameters (we use a vectorized implementation).
    state0_batch = []
    reward_batch = []
    action_batch = []
    terminal1_batch = []
    state1_batch = []
    for e in experiences:
        state0_batch.append(e.state0)
        state1_batch.append(e.state1)
        reward_batch.append(e.reward)
        action_batch.append(e.action)
        terminal1_batch.append(0. if e.terminal1 else 1.)

    # Prepare and validate parameters.
    state0_batch = agent.process_state_batch(state0_batch)
    state1_batch = agent.process_state_batch(state1_batch)
    terminal1_batch = np.array(terminal1_batch)
    reward_batch = np.array(reward_batch)
    return state0_batch, state1_batch, reward_batch, terminal1_batch


class WriterCallback(cbks.Callback):

    def __init__(self, model, log_dir='./logs'):
        super().__init__()
        self.model = model
        self.log_dir = log_dir
        self.writer = SummaryWriter(logdir=log_dir)
        self.reward = None
        self.max_reward = None

    def on_step_end(self, step, logs={}):
        self.reward = float(logs['info']['accumulated_reward'])
        self.max_reward = float(logs['info']['max_accumulated_reward'])

    def on_episode_end(self, episode, logs={}):

        metrics = {}
        for k, v in logs.items():
            metrics[k] = v

        # Metrics from model
        s0, s1, r, t = sample_batch(self.model, 32)
        q = self.model.model.predict([s0])

        metrics_values = self.model.model.evaluate([s0], y=[q])
        if not isinstance(metrics_values, list):
            metrics_values = [metrics_values,]


        metrics.update({n : v for n, v in zip(self.model.model.metrics_names, metrics_values)})

        # Rewards
        metrics['accumulated_reward'] = self.reward
        metrics['max_accumulated_reward'] = self.max_reward

        # Percentage reward
        pr = np.minimum(np.maximum(self.reward / (self.max_reward + 1e-3), 0), 1.)
        metrics['percentage_reward'] = pr


        lr_var = self.model.trainable_model.optimizer.optimizer.lr
        metrics['lr'] = K.get_value(lr_var)

        for k, v in metrics.items():
            v = np.array(v).astype('float64')
            self.writer.add_scalar(k, v, episode)

        self.writer.flush()

        if episode % 20 == 0:
            self.model.save_weights(f'{self.log_dir}/model_params.h5f', overwrite=True)



class LogWriterCallback(cbks.Callback):

    def __init__(self, log_dir='./logs'):
        super().__init__()
        logs_directory = f'{log_dir}'
        self.writer = SummaryWriter(logdir=logs_directory)
        self.reward = None
        self.max_reward = None


    def on_episode_end(self, episode, logs={}):

        for k, v in logs.items():
            if isinstance(v, dict):
                self.write_dictionary(k, v, episode)
                continue
            value = np.array(v).astype('float64')
            if len(value.shape) == 0:
                self.write_scalar(k, v, episode)
            elif len(value.shape) == 3 and value.shape[-1] == 4: # We assume it's an RGBA image
                self.write_image(k, v, episode)
        self.writer.flush()


    def write_image(self, name, value, episode):
        value = np.swapaxes(value, 0, 1)
        value = np.swapaxes(value, 0, -1)
        #value = tf.convert_to_tensor(value)
        self.writer.add_image(name, value, episode)


    def write_scalar(self, name, value, episode):
        value = np.array(value).astype('float64')
        self.writer.add_scalar(name, value, episode)


    def write_dictionary(self, name, value, episode):
        value = {a : np.array(b).astype('float64') for a, b in value.items()}
        self.writer.add_scalars(name, value, episode)



class Scheduler(cbks.Callback):

    def __init__(self, callback, episodes=10000, min_value=0, max_value=1, log_name=None):
        super().__init__()
        self.episodes = episodes
        self.callback = callback
        self.minmax = (min_value, max_value)
        self.name = log_name

    def on_episode_end(self, episode, logs={}):
        m, M = self.minmax
        v = m + min(1, episode/self.episodes) * (M-m)
        self.callback(v)
        if self.name is not None:
            logs[str(self.name)] = np.float(v)



class LearningRateScheduler(Scheduler):

    def __init__(self, optimizer, lr_0, lr_f, episodes):

        cbk = lambda x: backend.set_value(optimizer.lr, x)

        super().__init__(callback=cbk, episodes=episodes, min_value=lr_0, max_value=lr_f, log_name='lr')



class RewardsCallback(cbks.Callback):

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.rewards = np.zeros(self.env.unwrapped.n_colors)
        self.max_rewards = np.zeros(self.env.unwrapped.n_colors)
        self.success = True

    def on_episode_begin(self, episode, logs={}):
        self.rewards = np.zeros(self.env.unwrapped.n_colors)
        self.max_rewards = np.zeros(self.env.unwrapped.n_colors)
        self.success = True


    def on_step_end(self, step, logs={}):
        self.accumulate_rewards()
        self.accumulate_success()

    def on_episode_end(self, episode, logs={}):
        # Log metrics
        avg_rewards, r0, r_others = self.compute_average_rewards()
        logs['rewards%'] = {'c' : r0, 'others' : r_others}
        logs['accumulated_reward'] = {f'c-{i}' : r for i, r in enumerate(avg_rewards)}
        logs['max_accumulated_reward'] = {f'c-{i}' : r for i, r in enumerate(self.max_rewards)}
        logs['success'] = float(self.success)


    def compute_average_rewards(self):
        env = self.env.unwrapped
        r_c = env.reward_color
        avg_rewards = np.clip(self.rewards / (self.max_rewards + 1), a_min=0., a_max=1.)

        r0 = avg_rewards[r_c]

        r_others = np.copy(avg_rewards)
        r_others[r_c] = 0
        r_others = np.sum(r_others) / (len(r_others) - 1)
        return avg_rewards, r0, r_others


    def accumulate_success(self):
        env = self.env.unwrapped
        x, y = env.pos
        self.success = self.success and not env.check_obstacle(x, y)


    def accumulate_rewards(self):
        env = self.env.unwrapped
        n_colors = env.n_colors
        for c in range(n_colors):
            m = env.get_max_reward(c)
            r = env.get_reward(c)
            self.rewards[c] += np.float(r)
            self.max_rewards[c] = m



class HistoryAccumulator(cbks.Callback):

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.history = None
        self.mask = None
        self.steps = 0

    @property
    def walk(self):
        h = self.history[:,:-self.env.unwrapped.sight_distance+1]
        return h / np.sum(h)


    def on_train_begin(self, logs=None):
        self.steps = 0



    def on_step_end(self, step, logs={}):
        env = self.env.unwrapped
        x, y = env.pos

        if self.history is None:
            self.history = np.zeros(env.tunnel.shape[:-1])
        self.history[x, y] += 1

        if self.mask is None:
            self.mask = np.zeros(env.tunnel.shape[:-1])

        self.mask += env._render_mask
        self.steps += 1



    def on_episode_end(self, episode, logs={}):
        rgba = self.get_plot_matrix()
        logs['tunnel'] = self.get_plot_matrix(plot_walk=False, plot_saliency=False)
        logs['tunnel_walk'] = self.get_plot_matrix(plot_walk=True, plot_saliency=False)
        logs['tunnel_saliency'] = self.get_plot_matrix(plot_walk=False, plot_saliency=True)
        logs['tunnel_all'] = self.get_plot_matrix(plot_walk=True, plot_saliency=True)

    def get_initial_position(self):
        return 6, 0



    def get_plot_matrix(self, plot_walk=True, plot_saliency=True):
        tunnel = self.env.get_imshow_data(show_position=False)
        cmap = cm.get_cmap('Set1')
        colors = list(cmap.colors)
        #colors.reverse()
        colors[-1] = (0,0,0)
        cmap.colors = colors
        m0 = tunnel == 0
        m1 = tunnel > 0
        m_obstacles = tunnel == np.max(tunnel)
        tunnel[m1] -= 1
        tunnel[tunnel==np.max(tunnel)] += 3
        tunnel[m0] = len(colors) - 1

        rgba_tunnel = cm.get_cmap('Set1')(tunnel)
        rgba_tunnel0 = cm.get_cmap('tab10')(tunnel)

        rgba_tunnel[m_obstacles] = rgba_tunnel0[m_obstacles]

        walk = self.walk
        walk /= np.max(walk)

        rgba_walk = cm.get_cmap('cividis')(np.ones_like(walk))
        #rgba_walk[:] = 0
        #rgba_walk[:, :, 1] = 1
        rgba_walk[:,:,-1] = walk


        mask = np.copy(self.mask[:, :-self.env.unwrapped.sight_distance + 1])

        mask /= self.steps
        mask = (mask - np.min(mask))/(np.max(mask) - np.min(mask)) * 0.8
        mask = np.power(mask, 1/3)

        if np.max(mask) == np.min(mask):
            mask = np.zeros_like(mask)

        rgba_mask = np.zeros_like(rgba_walk)
        rgba_mask[:, :, 0] = 1
        rgba_mask[:, :, 1] = 1
        rgba_mask[:, :, 2] = 1
        rgba_mask[:, :, -1] = mask

        h,w, c = rgba_tunnel.shape
        # Use PIL to merge images
        image_A_convert = PImage.fromarray((rgba_tunnel*255).astype('uint8'))
        image_B_convert = PImage.fromarray((rgba_walk*255).astype('uint8'))
        image_C_convert = PImage.fromarray((rgba_mask*255).astype('uint8'))

        if plot_walk:
            image_A_convert.paste(image_B_convert, mask=image_B_convert.split()[3])
        if plot_saliency:
            image_A_convert.paste(image_C_convert, mask=image_C_convert.split()[3])
        image_A_convert = image_A_convert.resize((w*5, h*5), resample=PImage.BOX)

        rgba = np.asarray(image_A_convert)

        return image_A_convert


    def plot_overlay_tunnel(self):
        data = self.env.get_imshow_data(show_position=False)

        cmap = cm.get_cmap('Set1')
        colors = list(cmap.colors)
        #colors.reverse()
        colors[-1] = (0,0,0)
        cmap.colors = colors
        m0 = data == 0
        m1 = data > 0
        data[m0] = len(colors) - 1
        data[m1] -= 1

        fg = plt.figure()
        ax = fg.gca()

        data = self.get_plot_matrix()
        plot = ax.imshow(data)
        plt.pause(1e-3)
        return fg, ax


    def save_plot(self, filename):
        #matplotlib.use("pgf")
        #matplotlib.rcParams.update({
        #    "pgf.texsystem": "pdflatex",
        #    'font.family': 'serif',
        #    'text.usetex': True,
        #    'pgf.rcfonts': False,
        #})
        fig, ax = self.plot_overlay_tunnel()
        ax.set_axis_off()
        fig.add_axes(ax)
        fig.savefig(f'{filename}.png')
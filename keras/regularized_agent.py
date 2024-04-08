from collections import defaultdict
import numpy as np
import time
from tensorboardX import SummaryWriter
from interpretable.regularized_policy import RegularizedPolicy as BaseRegularizedAgent, boltzmann, egreedy



class RegularizedAgent(BaseRegularizedAgent):


    def __init__(self, wrapped_agent, env, num_masks=200, boltzmann_tau=1., max_num_steps=1e16):
        self.boltzmann_tau = boltzmann_tau
        super().__init__(
            q_agents=[wrapped_agent, ],
            num_actions=3,
            policy_fcn_regul=lambda x: boltzmann(x, self.boltzmann_tau),
            policy_fcn_result=lambda x: boltzmann(x, self.boltzmann_tau)
        )
        self.env = env
        self.max_num_steps = int(max_num_steps)
        self.num_masks = num_masks

        self._render = None


        self._callbacks = None



    def get_q_values(self, state, mask=None):
        if mask is not None:
            state = self.apply_mask(state, mask)
        states = np.stack([np.expand_dims(self.env.get_color_observation(state, c), 0) for c in range(self.env.n_colors)], 0)
        agent = self.agents[0]
        q_values = agent.compute_batch_q_values(states)
        return q_values


    @property
    def num_policies(self):
        return self.env.n_colors

    def create_masks(self, state):
        import itertools
        masks = []
        m0 = np.zeros_like(state['position'])
        sx, sy = m0.shape
        sm = 2
        for i, j in list(itertools.product(list(range(0,sx, sm)), list(range(0, sy, sm)))):
            m = np.copy(m0)
            m[i:i+sm,j:j+sm] = 1
            masks += m,
        masks = np.vstack(map(lambda x: np.expand_dims(x, 0), masks))
        return masks


    def create_masks_(self, state):
        masks = []
        m0 = np.zeros_like(state['position'])
        for i in range(self.num_masks):
            m = np.copy(m0)
            sx, sy = m.shape
            for j in range(1):
                x = np.random.randint(0, m.shape[0])
                y = np.random.randint(0, m.shape[1])
                m[max(x-2, 0):min(x+2, sx), max(y-2,0):min(y+2, sy)] = 1
            masks += m,
        masks = np.vstack(map(lambda x: np.expand_dims(x, 0), masks))
        return masks


    def apply_mask(self, state, mask):
        sc = {k : np.copy(v) for k, v in state.items()}
        sc['colors'] *= np.expand_dims(mask, -1)
        sc['obstacles'] *= mask
        return sc


    def test(self, num_episodes=1e16, regul_factor=1., mask_regul=0.05, boltzmann_tau=1., render=False, callbacks=None):
        env = self.env

        self.boltzmann_tau = boltzmann_tau

        self._render = render
        if callbacks is None:
            self._callbacks = []
        else:
            self._callbacks = list(callbacks)
        print('S', end='')

        results = defaultdict(lambda : list())


        self.call_callbacks(lambda cb: cb.on_train_begin, logs={})

        for e in range(int(num_episodes)):
            logs = {}
            state = env.unwrapped.reset()
            self.call_callbacks(lambda cb: cb.on_episode_begin, episode=e, logs={})
            self.render()
            rewards = np.zeros(self.env.n_colors)
            acc_legibility = 0

            for step in range(self.max_num_steps):
                self.call_callbacks(lambda cb: cb.on_step_begin, step=step, logs={})
                action = self.select_action(state, self.env.reward_color, regul_factor)

                mask = self.compute_explanation(state, action, self.env.reward_color, mask_regul)
                env.unwrapped.set_render_mask_sight_window(mask)


                legibility = .25
                next_state, reward, done, info = env.unwrapped.step(action)
                self.render()
                acc_legibility += legibility
                avg_legibility = acc_legibility / (step + 1)
                rewards += self.rewards()

                r0, r_other = self.compute_avg_rewards(rewards)
                logs.update({'average_legibility': avg_legibility,})



                self.call_callbacks(lambda cb: cb.on_step_end, step=step, logs=logs)
                if done:

                    results['c0'].append(r0)
                    results['other'].append(r_other)
                    results['legibility'].append(avg_legibility)
                    break
                state = next_state

            self.call_callbacks(lambda cb: cb.on_episode_end, episode=e, logs=logs)
            print('.', end='')
        print('E')

        results['regul_factor'].append([regul_factor,])
        _res = dict(results)
        for k, v in results.items():
            _res[k] = np.nanmean(np.array(v).flatten())
            print('{:} {:.2f}          '.format(k, _res[k]), end='')
        print()


        return _res



    def render(self):
        if self._render:
            self.env.render()

    def rewards(self):
        """

        :return: Reward ratio for all colors
        """
        r = np.array([self.env.get_reward(c) for c in range(self.env.n_colors)])
        m = np.array([self.env.get_max_reward(c) + 1 for c in range(self.env.n_colors)])
        return r/m

    def compute_avg_rewards(self, reward_arr):
        """

        :param reward_arr:
        :return:  (r_c, r_others)
        """
        r0 = reward_arr[self.env.reward_color]
        r_other = sum(reward_arr[1:]) / (len(reward_arr)-1)
        return np.array((r0, r_other))

    def call_callbacks(self, fcn, **kwargs):
        for cbk in self._callbacks:
            f = fcn(cbk)
            f(**kwargs)
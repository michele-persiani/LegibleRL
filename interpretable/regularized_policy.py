import numpy as np
from scipy.stats import norm
from scipy.special import softmax


def boltzmann(values, tau, clip=(-7e2,7e2)):
    values = np.array(values, dtype='float64')
    if len(values.shape) == 1:
        values = np.expand_dims(values, 0)
    v = np.exp(np.clip(values / tau, clip[0], clip[1]))
    v /= np.sum(v, axis=1, keepdims=True)
    return v


def egreedy(values, tau):
    ids_max = np.max(values, axis=1, keepdims=True) == values
    e = tau / (values.shape[1]-1)
    p_a = np.zeros_like(values)
    p_a[ids_max] = 1 - tau
    p_a[~ids_max] = e
    return p_a


class RegularizedPolicy:

    def __init__(self, q_agents, num_actions, policy_fcn_regul, policy_fcn_result):
        """

        :param q_agents: list of N agents. agent(state) should return ar array of 'num_actions' values
        :param num_actions: number of actions of the agents
        :param policy_fcn_regul: policy function used for regularization. policy_fcn(q_values) return an array of probabilities
        :param policy_fcn_result: policy function used for selecting actions from the regularized Q-values. policy_fcn(q_values) return an array of probabilities
        """
        self.agents = q_agents
        self.policy_fcn_regul = policy_fcn_regul
        self.policy_fcn_result = policy_fcn_result
        self.num_actions = num_actions


    def create_masks(self, state):
        """
        Create the masks for computing interpretability
        :param state: the state that will later be masked
        :return: a numpy array (N,...) with all of the N masks stacked
        """
        raise NotImplementedError


    def apply_mask(self, state, mask):
        """
        Apply a mask to the given state
        :param state: a numpy array for the state
        :param mask: a numpy array for the mask
        :return: a numpy array for the masked state of the same dimensions of 'state'
        """
        raise NotImplementedError


    def mask_loss(self, mask):
        """
        Computes the cost of using a given mask
        :param mask:
        :return:
        """
        return np.sum(mask > 0)

    @property
    def num_policies(self):
        return len(self.agents)

    def get_q_values(self, state):
        """
        Computes the Q-values from all the considered agents
        :param state:
        :return: An array (N,A) where N in the number of agents and A the number of actions
        """
        q = [ag(state) for ag in self.agents]
        q = np.vstack(q)
        return q


    def select_action(self, state, base_policy, regul_factor):

        # Unregularized policy
        q_values = self.get_q_values(state)
        p_a = self.policy_fcn_regul(q_values)

        #--- Legible policy
        p_a_regul = self.policy_fcn_regul(q_values)
        p_pi_regul = p_a_regul / np.sum(p_a_regul, 0, keepdims=True)


        q_values_regul = q_values + regul_factor * np.log(p_pi_regul)
        p_a_regul = self.policy_fcn_result(q_values_regul)


        #--- P(a|s) of legible policy
        a = np.random.choice(3, p=p_a_regul[base_policy, :])
        return a


    def compute_explanation(self, state, a, base_policy, mask_regul):
        e_masks = self.create_masks(state)
        masks_costs = np.array([mask_regul * self.mask_loss(m) for m in e_masks])

        # Unregularized policy
        q_values = self.get_q_values(state)
        p_a = self.policy_fcn_result(q_values)


        m0 = self.find_masks(e_masks, state, 0).reshape(-1,1)
        m1 = self.find_masks(e_masks, state, 1).reshape(-1,1)

        # Masks scores
        #s = self.get_p_pi_m_a(state, e_masks, base_policy, a)


        s = self.get_p_pi_m(state, e_masks, base_policy, a)

        try:
            mean, std = norm.fit(s)
            std = np.maximum(std, 1e-16)
        except:
            print('aaaa')

        #p_values = 2 * np.minimum(1 - norm.cdf(s.flatten(), mean, std), norm.cdf(s.flatten(), mean, std))
        p_values = 1 - norm.cdf(s.flatten(), mean, std)

        sel_m = (1-p_values) > 0.6#s > 0.5#
        sel_m = s > 0.6

        r_mask = np.sum(e_masks * np.expand_dims(sel_m, (1,2)), 0)

        if np.any(np.isnan(r_mask)):
            print('nan computing masks')

        return r_mask


# ---------

    def find_masks(self, masks, state, color):
        s = state['colors'][:, :, color]
        s = np.sum(masks * s, axis=(1,2))
        return s > 0


    def get_q_unmasked_masked(self, state, masks):
        q_values = np.zeros((len(masks), self.num_policies, self.num_actions))
        q_values_p = np.zeros((len(masks), self.num_policies, self.num_actions))
        for i, m in enumerate(masks):
            masked_state = self.apply_mask(state, (1-m))
            q_values[i, :]   = self.get_q_values(state)
            q_values_p[i, :] = self.get_q_values(masked_state)
        return q_values, q_values_p


    def get_p_pi_m_a(self, state, masks, base_policy, a):
        q_values, q_values_p = self.get_q_unmasked_masked(state, masks)
        scores = self.scores_delta_q(q_values, q_values_p)
        p_m_pi_a = scores / scores.sum(0, keepdims=True)

        # 1st order explanations
        p_a_a = self._action_proba(q_values)

        num = p_m_pi_a * p_a_a
        p_pi_m_a = num / np.sum(num, 1, keepdims=True)

        if np.any(np.isnan(p_m_pi_a)):
            print('nan computing masks scores')

        s = p_pi_m_a[:, base_policy, a].copy()

        return s


    def get_p_pi_m(self, state, masks, base_policy, a):
        q_values, q_values_p = self.get_q_unmasked_masked(state, masks)


        scores = self.scores_dkl(q_values, q_values_p)
        p_m = 1. / scores.shape[0]
        p_pi = 1. / scores.shape[1]
        p_pi_m = scores / np.sum(scores+1e-6, axis=1, keepdims=True)

        #p_m_pi_s = p_pi_m / (p_pi_m+1e-6).sum(0, keepdims=True)

        if np.any(np.isnan(p_pi_m)):
            print('nan computing masks scores')
        #p_pi_m = p_m_pi_s / np.sum(p_m_pi_s, 1, keepdims=True)

        s = p_pi_m[:, base_policy].copy()
        if np.any(np.isnan(s)):
            print('aaa')
        return s


    def scores_delta_q(self, q_v, q_v_m):
        s = (q_v - q_v_m)**2
        p_a = self._action_proba(q_v)
        p_a_r = self._action_proba(q_v_m)

        d_p = 1e2 * (p_a - p_a_r)
        s = np.exp(s)
        s /= np.sum(s + 1e-6)
        return s


    def scores_dkl(self, q_v, q_v_m):
        p_a = self._action_proba(q_v)
        p_a_r = self._action_proba(q_v_m)

        dkl = 1e2*np.sum(p_a * np.log(p_a / p_a_r), -1)
        #assert np.all(dkl >= 0), 'dkl is always > 0'
        dkl -= np.max(dkl)
        dkl = np.exp(dkl)
        dkl /= dkl.sum()
        return dkl



    def scores_saliency(self, q_v, q_v_m):
        p_a = self._action_proba(q_v)
        p_a_r = self._action_proba(q_v_m)

        p_diff = (p_a - p_a_r)#[:,:, sel_a]


        expand = lambda x: np.swapaxes(np.repeat(np.expand_dims(x, -1), x.shape[-1], -1), -1, -2)
        remove_identity = lambda x: (1-np.eye(x.shape[-1])) * x
        normalize = lambda x: x / np.sum(x, -1, keepdims=True)

        p_a_0   = normalize(remove_identity(expand(p_a)))
        p_a_r_0 = normalize(remove_identity(expand(p_a_r)))

        dkl = np.nansum(p_a_0 * np.log(p_a_0/p_a_r_0), -1)


        k = 1. / (1 + 10*dkl)

        s = 2*k*p_diff / (p_diff + k)

        s = np.exp(s*100)
        s += 1e-6

        return s




    def _action_proba(self, q_values):
        assert len(q_values.shape) == 3
        return self.policy_fcn_result(q_values.reshape(-1, self.num_actions)).reshape(-1, self.num_policies, self.num_actions)

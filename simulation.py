import numpy as np

#    0  right
#    1  left
#    2  up
#    3  down

class Simulator:

    def __init__(self, T, Q, policy):
        self.T = T
        self.Q = Q
        self.policy = policy

        self.s = None

    @property
    def size(self):
        return self.T.shape[:2]

    @property
    def S(self):
        A_size, S_size = self.size
        S = np.zeros(S_size)
        S[self.s] = 1
        return S


    def initialize(self, s):
        """

        :param s: Integer. State enumeration num
        :return:
        """
        self.s = s



    def step(self):
        assert self.s is not None, 'call initialize() before step()'
        s = self.s
        a = self.policy(self.Q, s)
        t = self.T[a, s,:]
        A_size, S_size = self.size
        self.s = np.random.choice(S_size, p=t)


    def make_trajectory_stat(self, s, goal_s, n=100, max_steps=1000):
        """

        :param s: initial state num
        :param goal_s: goal state num
        :param n: number of trajectories
        :param max_steps: max length of trajectories
        :return:
        """
        _, size = self.size
        result = np.zeros((n, size))
        for i in range(n):
            self.initialize(s)
            n_steps = 0
            while True:
                self.step()
                result[i, self.s] += 1
                n_steps += 1
                if (s == goal_s) or (n_steps >= max_steps):
                    break
            print('.', end='')
        print()
        return result



    @staticmethod
    def softmax_policy(a=1):
        def policy(Q, i, b):
            Q = Q[i, :]
            Q = Q.flatten()
            P = np.exp(b *Q)
            P /= np.sum(P)
            a = np.random.choice(len(Q), p=P)
            return a

        return lambda Q, s: policy(Q, s, a)


    @staticmethod
    def epsilon_greedy_policy(eps=1):
        def policy(Q, i, b):
            Q = Q[i, :]
            Q = Q.flatten()
            if np.random.random() < b:
                return np.random.choice(len(Q))
            else:
                return np.argmax(Q)

        return lambda Q, s: policy(Q, s, eps)


def get_trajectory(sim, i, g):
    traj = sim.make_trajectory_stat(i, g, max_steps=30, n=10)
    traj[traj > 1] = 1
    avg_T = np.sum(traj, 0)
    avg_T /= np.sum(avg_T)
    T = avg_T.reshape((size, size))
    return T

def get_goal_state(size, goals, trg_goal):

    grid = np.zeros((size, size))
    grid[goals[trg_goal][1], goals[trg_goal][0]] = 1

    G = grid.reshape(-1)
    g = np.argwhere(G).reshape(-1)
    return g


def softmax(Q, beta):
    Q = np.exp(beta * Q)
    Q /= np.sum(Q, 1, keepdims=True)
    return Q


def tree_search_a(T, entropies, s, n):
    best_children = []
    for a in range(T.shape[0]):
        ps_new = T[a, s, ...]
        s_new = np.random.choice(T.shape[-1], p=ps_new)
        best_children += tree_search(T, entropies, s_new, n),
    return np.array(best_children)



def tree_search(T, entropies, s, n):
    ce = entropies
    best_ce = np.min(ce[s,...])
    if n == 0:
        return best_ce

    h = []
    for i in range(T.shape[0]):
        ps_new = T[i, s, ...]
        s_new = np.random.choice(T.shape[-1], p=ps_new)
        hc = tree_search(T, entropies, s_new, 0)
        h += hc,
    i_best = np.argmin(h)
    ps_new = T[i_best, s, ...]
    s_new = np.random.choice(T.shape[-1], p=ps_new)
    ce_best_ch = tree_search(T, entropies, s_new, n-1)
    return best_ce + ce_best_ch
    best_children = []
    for i in range(T.shape[0]):
        ps_new = T[i, s, ...]
        s_new = np.random.choice(T.shape[-1], p=ps_new)
        best_children += tree_search(T, entropies, s_new, n-1),

    return best_ce + np.min(best_children, 0)



if __name__ == '__main__':
    from main import get_T_R, load_policies_q, cross_entropy, goals, size
    import matplotlib.pyplot as plt
    beta = 20
    beta_ce = 3
    reg_factor = 1

    trg_goal = 0
    discount_factor = 0.9
    policies_q = load_policies_q(discount_factors=(discount_factor,))

    ce = [cross_entropy(i, policies_q, beta=beta_ce)[discount_factor] for i in range(len(goals))]
    Q = policies_q[discount_factor]
    T, R = get_T_R(size, goal_cell=goals[trg_goal])



    def regularized_policy(Q, s):
        n = 10
        a = tree_search_a(T, ce[trg_goal], s, n)
        a -= np.mean(a)

        proba_base = softmax(Q, beta_ce)
        h_a =   np.log(proba_base)
        Q_ce = ce[trg_goal][s,...]

        Q_reg = h_a - reg_factor * a

        pi = softmax(Q_reg, beta)

        pi_sel = pi

        Pa = pi_sel[s,:]
        a = np.random.choice(len(Pa), p=Pa)
        return a


    sim     = Simulator(T, Q[trg_goal], Simulator.softmax_policy(a=beta))
    sim_reg = Simulator(T, Q[trg_goal], regularized_policy)


    g = get_goal_state(size, goals, trg_goal)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    ax1.set_title('Normal')
    ax2.set_title('Regularized')
    s = size**2-48
    for i in range(size**2-1, 0, -5): ##
        s = i
        traj = get_trajectory(sim, s, g)
        traj_reg = get_trajectory(sim_reg, s, g)

        ax1.imshow(traj)
        ax2.imshow(traj_reg)
        plt.pause(1)
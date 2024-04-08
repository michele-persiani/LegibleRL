import numpy as np
from mdptoolbox_git import mdp
from concurrent.futures import ThreadPoolExecutor
import itertools

def get_one_hot(size, pos, reshape=None):
    a = np.zeros(size)
    a[pos] = 1.
    if reshape is not None:
        a = a.reshape(reshape)
    return a


def find_x_y(s):
    x = np.argwhere(np.sum(s==1, 1)).flatten()[0]
    y = np.argwhere(np.sum(s==1, 0)).flatten()[0]
    return x, y


def get_P_A(X, dx, dy):
    """
    :param X: (S, S) matrix with a single cell equal 1
    :param dx: movement along x
    :param dy: movement along y
    :return: the transition matrix of size (S,S)
    """
    x, y, = find_x_y(X)
    xmax, ymax = X.shape
    P = np.zeros(X.shape)
    nx = max(0, min(xmax-1, x+dx))
    ny = max(0, min(ymax-1, y+dy))
    P[nx, ny] = 1.
    return P


def get_T_R(grid_size, goal_cell):
    T = np.zeros((4, grid_size**2, grid_size**2))
    R = np.zeros((grid_size, grid_size, 4))
    g_x, g_y = goal_cell

    # A, X, Y, X, Y = P
    action_dx_dy = {
        0 : (0, 1),  # right
        1 : (0, -1), # left
        2 : (-1, 0), # up
        3 : (1, 0)   # down
    }
    for i in range(4):
        dx, dy = action_dx_dy[i]
        for j in range(grid_size**2):
            s = get_one_hot(grid_size**2, j, reshape=(grid_size, grid_size))
            Pa = get_P_A(s, dx, dy)
            x, y = find_x_y(s)
            dest_x, dest_y = find_x_y(Pa)
            if dest_x == g_x and dest_y == g_y:
                R[x, y, i] = 1
            if dest_x == x and dest_y == y:
                R[x, y, i] = -1
            R[x, y, i] -= 0.02

            T[i, j, :] = Pa.flatten()

    R = R.reshape(-1, 4)

    return T, R


def get_V(pi, reshape=None):
    v = np.array(pi.V)
    if reshape is not None:
        v = v.reshape(reshape)
    return v

def compute_Pa(Q, beta=1.):
    pi = np.exp(beta * Q)
    pi /= np.sum(pi, 1, keepdims=True)
    return pi

def get_P_a(Q, beta=1.):
    Q = np.exp(beta*Q)
    Q /= np.sum(Q,1, keepdims=True)
    return Q

get_det_policy = lambda Q, size: np.argmax(Q, 1).reshape(size, size)


def save_q(id, Q, discount):
    filename = f'q/{id}_{discount}'
    np.save(filename, np.array(Q))

def load_q(id, discount):
    filename = f'q/{id}_{discount}.npy'
    return np.load(filename)


def train_policy(grid_size, goal_cell, n_iter, discount):
    T, R = get_T_R(grid_size, goal_cell=goal_cell)
    pi = mdp.QLearning(T, R, discount, n_iter=n_iter, learning_rate=0.1, reset_every=10, policy_epsilon=.4)
    pi.run()
    return pi.Q

def make_policies_q(goals, size, n_iter, discount_factors=(0.8, 0.85, 0.9, 0.95)):

    results = []
    for df in discount_factors:
        print(f'-------- Make discount factor {df} --------')
        with ThreadPoolExecutor(max_workers=10) as executor:
            pi = [executor.submit(train_policy, size, g, n_iter, df) for g in goals]
            results += [[df,] + pi,]

    for r in results:
        df = r[0]
        policies = r[1:]
        for i, p in enumerate(policies):
            save_q(i, p.result(), df)



def load_policies_q(discount_factors=(0.8, 0.85, 0.9, 0.95)):
    res = {}
    for df in discount_factors:
        i = 0
        policies = []
        while True:
            try:
                policies.append(load_q(i, df))
            except FileNotFoundError as e:
                break
            i += 1
        res[df] = policies

    return res



def cross_entropy(pi_id, policies, beta=1.5):
    res = {}
    for df in policies.keys():
        q = policies[df]
        p = np.vstack([np.expand_dims(get_P_a(i, beta=beta), 0) for i in q])
        res[df] = -np.log(p[pi_id, ...])  + np.log(np.mean(p, 0)) - np.log(1/p.shape[0])
    return res


#    0  right
#    1  left
#    2  up
#    3  down

def print_policy(p):
    from collections import defaultdict
    to_str = lambda i: defaultdict(lambda : 'X', {0 : 'R', 1: 'L', 2: 'U', 3: 'D'})[i]

    for row in p:
        for cell in row:
            print(to_str(cell)+'  ', end='')
        print()




goals = [(3,8), (3,12)]
size= 20


if __name__ == '__main__':

    np.set_printoptions(suppress=True, precision=3)
    n_iter=70000000


    make_policies_q(goals, size, n_iter, discount_factors=(.9,))
    exit(0)

    # Use simulator.py instead

    policies_q = load_policies_q(discount_factors=(.9,))

    beta=1
    ce_0 = cross_entropy(0, policies_q, beta=beta)
    ce_1 = cross_entropy(1, policies_q, beta=beta)
    #ce_2 = cross_entropy(2, policies_q, beta=beta)

    for df, reg_factor in itertools.product(policies_q.keys(), (1., .9, .7, .5, .3)):
        pi0, pi1 = policies_q[df]
        reg_0 = ce_0[df]
        reg_1 = ce_1[df]
        #reg_2 = ce_2[df]

        p_a_0 = get_P_a((1-reg_factor) * pi0 - reg_factor * reg_0)
        p_a_1 = get_P_a((1-reg_factor) * pi1 - reg_factor * reg_1)
        #p_a_2 = get_P_a((1-reg_factor) * pi2 - reg_factor * reg_2)

        pi  = [pi0, pi1,]
        p_a = [p_a_0, p_a_1,]

        print(f'-------- Discount factor {df} Regul factor {reg_factor} --------')
        i = 1
        P0 = get_det_policy(pi[i], size)
        P0[goals[i][0], goals[i][1]] = 10
        print_policy(P0)
        print()
        P_A = get_det_policy(p_a[i], size)
        P_A[goals[i][0], goals[i][1]] = 10
        print_policy(P_A)
        print()
        print()

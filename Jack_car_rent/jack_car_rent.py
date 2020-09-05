import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import math
import multiprocessing as mp
import itertools
"""

The stepped curves are showing the contours of the different policy actions, as a map over the state space.
They are a choice of visualisation of the policy, which has 441 states, and would not look quite so intuitive
listed as a table.

The numbers are the number of cars that the policy decides to move from first location to second location.
You can look up the optimal action from the 𝜋4 graph for a specific number of cars at each location by
finding the grid point (𝑛2,𝑛1) for it (reading horizontal axis first) and seeing what the number is inside
that area - move that number of cars from first to second location.

The final image shows the state value function of the optimal policy as a 3D surface with the base being
 the state and the height being the value.


즉, -5 ~ 5는 state value 나타낸 게 아니라 밤중에 몇 대를 옮길 지에 대한 action 을 나타냄.
n은 일정 시간 내 사건이 발생한 횟수. lambda는 그 n의 평균.

"""


MAX_CAR = 20
MOVE_CAR_ACTION_MAX = 5
AVG_FIRST_LOC_REQ = 3
AVG_SECOND_LOC_REQ = 4
AVG_FIRST_LOC_RTN = 3
AVG_SECOND_LOC_RTN = 2
DISCOUNT_RATE = 0.9
REWARD_RENT = 8
REWARD_MOVE = -2

RENT_LIM = 11
CONVERGE_THRESH = 0.05

poisson_cache = dict()


def poisson(n, Lambda):
    global RENT_LIM

    Pr = math.exp(-Lambda) * math.pow(Lambda, n) / math.factorial(n)
    if RENT_LIM < n:
        return 0
    else:
        return Pr
"""

def poisson(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache.keys():
        poisson_cache[key] = math.exp(-lam) * math.pow(lam, n) / math.factorial(n)
    return poisson_cache[key]
"""

class policy_iteration:
    def __init__(self, rent_lim, converge_thresh, discount_rate):
        self.actions    = np.arange(-MOVE_CAR_ACTION_MAX, MOVE_CAR_ACTION_MAX+1)
        self.policy    = np.zeros((MAX_CAR + 1, MAX_CAR + 1))
        self.values = np.zeros((MAX_CAR + 1, MAX_CAR + 1))
        self.inverse_actions = {act: idx for idx, act in enumerate(self.actions)}
        self.CONVERGE_THRESH = converge_thresh
        self.DISCOUNT_RATE   = discount_rate
        self.RENT_LIM        = rent_lim



    def solve(self):
        iterations = 0
        while True:
            self.values = self.policyEval(self.values, self.policy)
            policy_change, self.policy = self.policyImprove(self.actions, self.values, self.policy)

            if policy_change == 0:
                break
            iterations += 1



    def state_val_update(self, action, state, values):

        updated_state_val = REWARD_MOVE * abs(action)

        for rent1 in range(0, RENT_LIM):
            for rent2 in range(0, RENT_LIM):


                num_car_1st = int(min(state[0] - action, MAX_CAR))
                num_car_2nd = int(min(state[1] + action, MAX_CAR))

                RENT1 = min(num_car_1st, rent1)
                RENT2 = min(num_car_2nd, rent2)

                Pr_rent = poisson(rent1, AVG_FIRST_LOC_REQ) * poisson(rent2, AVG_SECOND_LOC_REQ)
                num_car_1st = num_car_1st - RENT1
                num_car_2nd = num_car_2nd - RENT2

                Rsa = (RENT1 + RENT2) * REWARD_RENT


                for rtn1 in range(0, RENT_LIM):
                    for rtn2 in range(0, RENT_LIM):

                        Pr_rtn = poisson(rtn1, AVG_FIRST_LOC_RTN) * poisson(rtn2, AVG_SECOND_LOC_RTN) * Pr_rent
                        num_car_1st_if = min(num_car_1st + rtn1, MAX_CAR)
                        num_car_2nd_if = min(num_car_2nd + rtn2, MAX_CAR)

                        updated_state_val += Pr_rtn * (Rsa + self.DISCOUNT_RATE * values[num_car_1st_if, num_car_2nd_if])

        return updated_state_val






    def expected_return_pe(self, policy, values, state):
        action = policy[state[0], state[1]]
        expected_state_val = self.state_val_update(action, state, values)
        return expected_state_val, state[0], state[1]





    def expected_return_pi(self, values, action, state):
        loc1_car_avail = state[0] >= action >= 0
        loc2_car_avail = action < 0 and state[1] >= abs(action)

        if (loc1_car_avail or loc2_car_avail) == False:
            return -float('inf'), state[0], state[1], action

        expected_state_val = self.state_val_update(action, state, values)
        return expected_state_val, state[0], state[1], action





    def policyEval(self, values, policy):

        global MAX_CAR


        while True:
            new_values = np.copy(values)
            k = np.arange(MAX_CAR + 1)
            state_table = ((i, j) for i, j in itertools.product(k, k))


            with mp.Pool(processes=8) as p:
                val_from_state = partial(self.expected_return_pe, policy, values)
                results = p.map(val_from_state, state_table)

            for value, loc1_num, loc2_num in results:
                new_values[loc1_num, loc2_num] = value


            difference = np.abs(new_values - values).sum()
            print('Distance from convergence : ', difference)

            values = new_values
            if difference < self.CONVERGE_THRESH:
                print('Complete convergence!')
                return values




    def policyImprove(self, actions, values, policy):
        new_policy = np.copy(policy)


        expected_action = np.zeros((MAX_CAR + 1, MAX_CAR + 1, np.size(actions)))
        val_from_act = dict()

        with mp.Pool(processes=8) as p:
            for action in actions:
                k = np.arange(MAX_CAR + 1)
                state_table = ((i, j) for i, j in itertools.product(k, k))

                val_from_act[action] = partial(self.expected_return_pi, values, action)
                results = p.map(val_from_act[action], state_table)
                for value, loc1_num, loc2_num, act in results:
                    expected_action[loc1_num, loc2_num, self.inverse_actions[act]] = value

        for i in range(expected_action.shape[0]):
            for j in range(expected_action.shape[1]):
                new_policy[i, j] = actions[np.argmax(expected_action[i, j])]

        policy_change = (new_policy != policy).sum()
        print('# Policy changed : ', policy_change)
        return policy_change, new_policy






    def plot(self):
        print(self.policy)
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        ax.imshow(self.policy, interpolation='nearest')
        plt.show()






if __name__ == '__main__':
    Jack = policy_iteration(rent_lim=RENT_LIM, converge_thresh=CONVERGE_THRESH, discount_rate=DISCOUNT_RATE)
    Jack.solve()

import numpy as np
import matplotlib.pyplot as plt
from random import*
import random



class Dyna_Q:

    # 1. 문제 설정
    def __init__(self, start, gamma, alpha, episodes, planning):
        self.start = start


        self.y = start[1]
        self.x = start[0]
        self.act_idx = 0

        self.new_x = self.x
        self.new_y = self.y
        self.new_act_idx = self.act_idx


        self.gamma = gamma
        self.alpha = alpha
        self.ep = episodes
        self.planning = planning

        self.actions = ["left", "right", "up", "down"]


        self.reward_map = np.array([[-1, -1,  -1, -1, -1,  -1, -1,   0,  1],
                                    [-1, -1,   0, -1, -1,  -1, -1,   0, -1],
                                    [-1, -1,   0, -1, -1,  -1, -1,   0, -1],
                                    [-1, -1,   0, -1, -1,  -1, -1,  -1, -1],
                                    [-1, -1,  -1, -1, -1,   0, -1,  -1, -1],
                                    [-1, -1,  -1, -1, -1,  -1, -1,  -1, -1]])

        self.action_val = np.zeros((len(self.reward_map), len(self.reward_map[0]), self.actions.__len__()))
        self.MODEL      = np.empty((len(self.reward_map), len(self.reward_map[0]), self.actions.__len__()), dtype=object)





    def Take_action(self, direct):

        max_y = len(self.reward_map) - 1
        max_x = len(self.reward_map[0]) - 1

        if direct == "left":
            self.new_x = self.x - 1
            self.new_y = self.y
            if self.new_x < 0:
                self.new_x = 0
            elif self.reward_map[self.new_y][self.new_x] == 0:
                print((self.new_y,self.new_x))
                self.new_x = self.x

        elif direct == "right":
            self.new_x = self.x + 1
            self.new_y = self.y
            if self.new_x > max_x:
                self.new_x = max_x
            elif self.reward_map[self.new_y][self.new_x] == 0:
                print((self.new_y,self.new_x))
                self.new_x = self.x

        elif direct == "up":
            self.new_x = self.x
            self.new_y = self.y + 1
            if self.new_y > max_y:
                self.new_y = max_y
            elif self.reward_map[self.new_y][self.new_x] == 0:
                print((self.new_y,self.new_x))
                self.new_y = self.y

        elif direct == "down":
            self.new_x = self.x
            self.new_y = self.y - 1
            if self.new_y < 0:
                self.new_y = 0
            elif self.reward_map[self.new_y][self.new_x] == 0:
                print((self.new_y,self.new_x))
                self.new_y = self.y




    def Simple_Maze_QL(self):


        EP_log = []
        returns_log = []
        step_log = []


        for i in range(self.ep):

            self.x = self.start[0]
            self.y = self.start[1]
            returns = 0
            step = 0
            path = [[self.y, self.x]]

            while 1:

                self.act_idx = self.epsilon_greedy_policy("QL")

                direct = self.actions[self.act_idx]
                self.Take_action(direct)

                reward = self.reward_map[self.new_y][self.new_x]

                self.update_action_QL(reward)

                returns += reward
                if i == (self.ep - 1):
                    #path.append(direct)
                    path.append([self.y, self.x])

                if reward == 1 or reward == -100:
                    break

                step += 1

            print("END episode", i + 1, " epsilon = ", float(1 / (i + 1)), "\n")

            if i % 10 == 0:
                EP_log.append(i)
                returns_log.append(returns)


            step_log.append(step)

        print(np.round(self.action_val, 3))
        print(path)
        return path, returns_log,  EP_log, step_log






    def update_action_QL(self, R):
        action_val = self.action_val[self.y][self.x][self.act_idx]

        possible_act_vals = self.action_val[self.new_y][self.new_x]
        MAX_new_action_val = self.action_val[self.new_y][self.new_x][np.argmax(possible_act_vals)]

        self.action_val[self.y][self.x][self.act_idx] += self.alpha * (R + self.gamma * MAX_new_action_val - action_val)

        self.x = self.new_x
        self.y = self.new_y






    def epsilon_greedy_policy(self, mode):

        possible_act_vals = []

        # Learning mode 에 따라, A를 S로 부터 구할 지, S'으로 부터 구할 지 결정..
        if mode == "Sarsa":
            possible_act_vals = self.action_val[self.new_y][self.new_x]
        elif mode == "QL":
            possible_act_vals = self.action_val[self.y][self.x]



        actions = self.actions


        # 입실론-greedy 가 GLIE --> greedy imp 는 1 - 1/k, random imp(epsilon) 는 1/mk 확률.
        epsilon = 0.1
        greedy_prob = 1 - epsilon


        # 입실론-greedy improvement 적용. max_action 바로 뽑을수도, 그냥 random 으로 뽑아야 될 수도있다.
        val_max_idx    = np.argmax(possible_act_vals)
        max_action     = actions[val_max_idx]
        chosen_action  = np.random.choice([max_action, actions], p=[greedy_prob, epsilon])


        #  random 으로 greedy 가 될수도 있다.
        if chosen_action == max_action:
            return val_max_idx
        else:
            random_idx = np.random.choice(range(len(actions)))
            return random_idx






# MODEL 용 함수.


    def Simple_Maze_DynaQ(self):
        number_rows = len(self.reward_map)
        number_cols = len(self.reward_map[0])
        number_acts = len(self.actions)

        self.action_val = np.zeros((number_rows, number_cols, number_acts))
        self.x = self.start[0]
        self.y = self.start[1]

        observed_state  = []
        observed_action = np.zeros((number_rows, number_cols, number_acts))

        iteration = 0

        while iteration < 2000:
                                                                                      # (a) state 는 current state 상태.
            self.act_idx = self.epsilon_greedy_policy("QL")                           # (b) epsilon-greedy 로 action 고르기.

            if [self.y, self.x] not in observed_state:
                observed_state.append([self.y, self.x])                               # observed state & action 에 기록.
            observed_action[self.y][self.x][self.act_idx] = 1


            direct = self.actions[self.act_idx]                                       # (c) action 을 취하고, R, S' 확인.
            self.Take_action(direct)
            reward = self.reward_map[self.new_y][self.new_x]

            self.MODEL[self.y][self.x][self.act_idx] = (self.new_y, self.new_x, reward)                          # (d) MODEL 에 전달.

            self.update_action_QL(reward)                                             # (e) Q(S,A) update.


            for i in range(self.planning):
                y, x    = self.Random_observed_state(observed_state)                          # (f) random 으로, 이전에 관찰되었던 state 'S'
                act_idx = self.Random_observed_action(observed_action[y][x])                  # (g) 'S'에서 이전에 취했었던 'A' 고르기.

                (new_y, new_x, reward) = self.MODEL[y][x][act_idx]

                self.update_action_QL_MODEL(y, x, new_y, new_x, act_idx, reward)


            print("\n\n\niteration :", iteration)
            iteration += 1










        EP_log = []
        returns_log = []
        step_log = []


        for i in range(self.ep):

            self.x = self.start[0]
            self.y = self.start[1]
            returns = 0
            step = 0
            path = [[self.y, self.x]]

            while 1:

                self.act_idx = self.epsilon_greedy_policy("QL")

                direct = self.actions[self.act_idx]
                self.Take_action(direct)

                reward = self.reward_map[self.new_y][self.new_x]

                self.update_action_QL(reward)

                returns += reward
                if i == (self.ep - 1):
                    #path.append(direct)
                    path.append([self.y, self.x])

                if reward == 1 or reward == -100:
                    break

                step += 1

            print("END episode", i + 1, " epsilon = ", float(1 / (i + 1)), "\n")

            if i % 10 == 0:
                returns_log.append(returns)


            step_log.append(step)
            EP_log.append(i)

        print(path)
        return path, returns_log,  EP_log, step_log








    def epsilon_greedy_policy_MODEL(self, y, x):

        possible_act_vals = self.action_val[y][x]

        actions = self.actions

        epsilon = 0.1
        greedy_prob = 0.9

        val_max_idx    = np.argmax(possible_act_vals)
        max_action     = actions[val_max_idx]
        chosen_action  = np.random.choice([max_action, actions], p=[greedy_prob, epsilon])

        if chosen_action == max_action:
            return val_max_idx
        else:
            random_idx = np.random.choice(range(len(actions)))
            return random_idx








    def Random_observed_state(self, observed_state):
        state_idx = randint(0, len(observed_state) - 1)
        y = observed_state[state_idx][0]
        x = observed_state[state_idx][1]
        return y, x


    def Random_observed_action(self, observed_action_in_state):
        act_idx = randint(0, len(self.actions) - 1)
        while observed_action_in_state[act_idx] == 0:
            act_idx = randint(0,  len(self.actions) - 1)
            print(observed_action_in_state)
        return act_idx




    def Take_action_MODEL(self, y, x, direct):

        max_y = len(self.reward_map) - 1
        max_x = len(self.reward_map[0]) - 1

        new_x = 0
        new_y = 0

        if direct == "left":
            new_x = x - 1
            new_y = y
            if new_x < 0:
                new_x = 0
            elif self.reward_map[new_y][new_x] == 0:
                new_x = x

        elif direct == "right":
            new_x = x + 1
            new_y = y
            if new_x > max_x:
                new_x = max_x
            elif self.reward_map[new_y][new_x] == 0:
                new_x = x

        elif direct == "up":
            new_x = x
            new_y = y + 1
            if new_y > max_y:
                new_y = max_y
            elif self.reward_map[new_y][new_x] == 0:
                new_y = y

        elif direct == "down":
            new_x = x
            new_y = y - 1
            if new_y < 0:
                new_y = 0
            elif self.reward_map[new_y][new_x] == 0:
                new_y = y


        return new_y, new_x



    def update_action_QL_MODEL(self, y, x, new_y, new_x, act_idx, R):
        action_val = self.action_val[y][x][act_idx]

        MAX_new_action_val = np.amax(self.action_val[new_y][new_x])
        self.action_val[y][x][act_idx] += self.alpha * (R + self.gamma * MAX_new_action_val - action_val)





def plot(EP, Q1, Q2, Q3):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    graph = ax.plot(EP, Q1, EP, Q2, EP, Q3)
    ax.legend(graph, ['n=0', 'n=5', 'n=50'], loc=0)
    ax.set_ylabel("Steps per episode")
    ax.set_xlabel("episode")

    axes = plt.gca()
    axes.set_ylim([0, 800])

    plt.show()




problem = Dyna_Q(start=[0,2], gamma=0.95, alpha=0.1, episodes=350, planning=5)
Q_path, Q_rewards, Q_EP, Q_step = problem.Simple_Maze_QL()
Q_path2, Q_rewards2, Q_EP2, Q_step2 = problem.Simple_Maze_DynaQ()

problem2 = Dyna_Q(start=[0,2], gamma=0.95, alpha=0.1, episodes=350, planning=50)
Q_path3, Q_rewards3, Q_EP3, Q_step3 = problem.Simple_Maze_DynaQ()


print("\n\n")
print("Q_path\n" , Q_step.__len__())
print("Q_path2\n" , Q_step2.__len__())
print("Q_path3\n" , Q_step3.__len__())
plot(Q_EP3, Q_step, Q_step2, Q_step3)


import numpy as np
import matplotlib.pyplot as plt
import random



class CliffWalking:

    # 1. 문제 설정
    def __init__(self, start, gamma, alpha, episodes):
        self.start = start

        self.x = start[0]
        self.y = start[1]
        self.act_idx = 0

        self.new_x = self.x
        self.new_y = self.y
        self.new_act_idx = self.act_idx


        self.gamma = gamma
        self.alpha = alpha
        self.ep = episodes

        self.actions = ["left", "right", "up", "down"]


        self.reward_map = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, -100, -100, -100, -100, -100, -100, -100, -100,  1]])

        self.action_val = np.zeros((len(self.reward_map), len(self.reward_map[0]), self.actions.__len__()))



    def Take_action(self, direct):

        max_y = len(self.reward_map) - 1
        max_x = len(self.reward_map[0]) - 1

        if direct == "left":
            self.new_x = self.x - 1
            self.new_y = self.y
            if self.new_x < 0:
                self.new_x = 0

        elif direct == "right":
            self.new_x = self.x + 1
            self.new_y = self.y
            if self.new_x > max_x:
                self.new_x = max_x

        elif direct == "up":
            self.new_x = self.x
            self.new_y = self.y + 1
            if self.new_y > max_y:
                self.new_y = max_y

        elif direct == "down":
            self.new_x = self.x
            self.new_y = self.y - 1
            if self.new_y < 0:
                self.new_y = 0




    def Cliff_walk_result_QL(self):

        path = [[self.x, self.y]]
        EP_log = []
        returns_log = []

        for i in range(self.ep):

            self.x = self.start[0]
            self.y = self.start[1]
            self.act_idx = self.epsilon_greedy_policy(i + 1, "QL")
            returns = 0



            while 1:

                self.act_idx = self.epsilon_greedy_policy(i + 1, "QL")

                direct = self.actions[self.act_idx]
                self.Take_action(direct)

                reward = self.reward_map[self.new_y][self.new_x]

                self.update_action_QL(reward)

                returns += reward
                if i == (self.ep - 1):
                    path.append(direct)
                    path.append([self.x, self.y])

                if reward == 1 or reward == -100:
                    break

            print("END episode", i + 1, " epsilon = ", float(1 / (i + 1)), "\n")

            if i % 10 == 0:
                EP_log.append(i)
                returns_log.append(returns)

        print(np.round(self.action_val, 3))
        print(path)
        print(returns_log)
        return path, returns_log,  EP_log







    def Cliff_walk_result_Sarsa(self):

        path = [[self.x, self.y]]
        EP_log = []
        returns_log = []

        # 2. 모든 에피소드 돌리기.
        for i in range(self.ep):

            self.x = self.start[0]
            self.y = self.start[1]
            self.act_idx = self.epsilon_greedy_policy(i + 1, "Sarsa")
            returns = 0

            while 1:

                direct = self.actions[self.act_idx]

                self.Take_action(direct)
                reward = self.reward_map[self.new_y][self.new_x]


                self.new_act_idx = self.epsilon_greedy_policy(i + 1, "Sarsa")
                self.update_action_Sarsa(reward)


                returns += reward

                if i == (self.ep - 1):
                    path.append(direct)
                    path.append([self.x, self.y])

                if reward == 1 or reward == -100:
                    break

            print("END episode", i + 1, " epsilon = ", float(1 / (i + 1)), "\n")
            if i % 10 == 0:
                EP_log.append(i)
                returns_log.append(returns)

        print(np.round(self.action_val, 3))
        print(path)
        print(returns_log)

        return path[2:], returns_log, EP_log











    def epsilon_greedy_policy(self, curr_ep, mode):

        possible_act_vals = []

        # Learning mode 에 따라, A를 S로 부터 구할 지, S'으로 부터 구할 지 결정..
        if mode == "Sarsa":
            possible_act_vals = self.action_val[self.new_y][self.new_x]
        elif mode == "QL":
            possible_act_vals = self.action_val[self.y][self.x]



        actions = self.actions


        # 입실론-greedy 가 GLIE --> greedy imp 는 1 - 1/k, random imp(epsilon) 는 1/mk 확률.
        epsilon = float(1.0/curr_ep)
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






    def update_action_Sarsa(self, R):
        action_val = self.action_val[self.y][self.x][self.act_idx]
        new_action_val = self.action_val[self.new_y][self.new_x][self.new_act_idx]

        self.action_val[self.y][self.x][self.act_idx] += self.alpha * (R + self.gamma * new_action_val - action_val)
        self.x = self.new_x
        self.y = self.new_y
        self.act_idx = self.new_act_idx



    def update_action_QL(self, R):
        action_val = self.action_val[self.y][self.x][self.act_idx]

        possible_act_vals = self.action_val[self.new_y][self.new_x]
        MAX_new_action_val = self.action_val[self.new_y][self.new_x][np.argmax(possible_act_vals)]

        self.action_val[self.y][self.x][self.act_idx] += self.alpha * (R + self.gamma * MAX_new_action_val - action_val)
        self.x = self.new_x
        self.y = self.new_y






def plot(EP, R_Sarsa, R_QL):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    graph = ax.plot(EP, R_Sarsa, EP, R_QL)
    ax.legend(graph, ['Sarsa', 'QL'], loc=0)
    ax.set_ylabel("Reward per episode")
    ax.set_xlabel("episode")

    axes = plt.gca()
    axes.set_ylim([-100, 0])

    plt.show()








problem = CliffWalking(start=[0,3], gamma=1, alpha=0.03, episodes=1001)
Sa_path, Sa_rewards, Sa_EP = problem.Cliff_walk_result_Sarsa()

problem2 = CliffWalking(start=[0,3], gamma=1, alpha=0.03, episodes=1001)
Q_path, Q_rewards, Q_EP = problem2.Cliff_walk_result_QL()

print("\n\n")
print("Sa_path", Sa_path)
print("Q_path" , Q_path)
plot(Sa_EP, Sa_rewards, Q_rewards)


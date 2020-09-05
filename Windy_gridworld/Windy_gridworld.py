import numpy as np
import matplotlib.pyplot as plt
import random

class WindyGridWorld:

    # 1
    def __init__(self, start, gamma, alpha, episodes, Mode):
        self.x = start[0]
        self.y = start[1]
        self.act_idx = random.randint(0,3)
        self.new_x = self.x
        self.new_y = self.y
        self.new_act_idx = 0


        self.map_r = 7
        self.map_c = 10
        self.goal_y = 3
        self.goal_x = 7

        if Mode == "Standard":
            self.actions = ["left", "right", "up", "down"]
        elif Mode == "King":
            self.actions = ["left", "right", "up", "down", "left-up", "left-down", "right-up", "right-down"]


        self.action_val = np.zeros((self.map_r, self.map_c, self.actions.__len__()))
        self.reward_map = np.full((self.map_r, self.map_c), -1)
        self.reward_map[self.goal_y][self.goal_x] = 1
        self.wind   = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        self.gamma = gamma
        self.alpha = alpha
        self.ep = episodes





    def Take_action(self, direct, max_x, max_y):
        if direct == "left":
            new_x = self.x - 1
            if new_x < 0:
                new_x = 0

            new_y = self.y + self.wind[new_x]
            if new_y > max_y:
                new_y = max_y

            self.new_x = new_x
            self.new_y = new_y



        elif direct == "right":
            new_x = self.x + 1
            if new_x > max_x:
                new_x = max_x

            new_y = self.y + self.wind[new_x]
            if new_y > max_y:
                new_y = max_y

            self.new_x = new_x
            self.new_y = new_y



        elif direct == "up":
            new_x = self.x
            new_y = self.y + 1 + self.wind[self.new_x]
            if new_y > max_y:
                new_y = max_y

            self.new_x = new_x
            self.new_y = new_y  # map의 row[0]가 맨위부터 시작해서.....



        elif direct == "down":
            new_x = self.x
            new_y = self.y - 1 + self.wind[self.new_x]

            if new_y > max_y:
                new_y = max_y
            elif self.y - 1 < 0:
                new_y = 0

            self.new_x = new_x
            self.new_y = new_y



        elif direct == "left-up":
            new_x = self.x - 1
            if new_x < 0:
                new_x = 0

            new_y = self.y + 1 + self.wind[new_x]
            if new_y > max_y:
                new_y = max_y

            self.new_x = new_x
            self.new_y = new_y


        elif direct == "left-down":
            new_x = self.x - 1
            if new_x < 0:
                new_x = 0

            new_y = self.y - 1 + self.wind[self.new_x]
            if new_y > max_y:
                new_y = max_y
            elif self.y - 1 < 0:
                new_y = 0

            self.new_x = new_x
            self.new_y = new_y


        elif direct == "right-up":
            new_x = self.x + 1
            if new_x > max_x:
                new_x = max_x

            new_y = self.y + 1 + self.wind[new_x]
            if new_y > max_y:
                new_y = max_y

            self.new_x = new_x
            self.new_y = new_y



        elif direct == "right-down":
            new_x = self.x + 1
            if new_x > max_x:
                new_x = max_x

            new_y = self.y - 1 + self.wind[self.new_x]
            if new_y > max_y:
                new_y = max_y
            elif self.y - 1 < 0:
                new_y = 0

            self.new_x = new_x
            self.new_y = new_y





    def Windy_move(self):

        path = []
        time_step_log = []
        EP_log = []

        max_x = self.map_c - 1
        max_y = self.map_r - 1



        # 2
        for i in range(self.ep):
            self.x = 0
            self.y = 3
            self.new_x = self.x
            self.new_y = self.y


            self.act_idx = random.randint(0,3)
            time_step = 0

            # 3
            while 1:


                direct = self.actions[self.act_idx]

                # 4
                self.Take_action(direct, max_x, max_y)
                reward = self.reward_map[self.new_y][self.new_x]

                # 5
                self.new_act_idx = self.epsilon_greedy_policy(i+1)

                # 6
                #print(time_step, direct, self.x, self.y)
                self.update_action_Sarsa(reward)

                if i == (self.ep - 1):
                    path.append(direct)



                time_step += 1
                if self.x == self.goal_x and self.y == self.goal_y:
                    print(time_step)
                    break



            print("END episode", i+1)
            print("time_step", time_step)


            time_step_log.append(time_step)
            EP_log.append(i)

            print("epsilon = ", float(1/(i+1)))
            print("\n")

        print(np.round(self.action_val, 3))
        print(path)

        return path, time_step_log, EP_log





    def update_action_Sarsa(self, R):

        #print("before =", self.x, self.y, self.act_idx)
        #print("new = ", self.new_x, self.new_y, self.new_act_idx)
        #print("\n")

        action_val     = self.action_val[self.y][self.x][self.act_idx]
        new_action_val = self.action_val[self.new_y][self.new_x][self.new_act_idx]


        self.action_val[self.y][self.x][self.act_idx] += self.alpha * (R + self.gamma * new_action_val - action_val)
        self.x = self.new_x
        self.y = self.new_y
        self.act_idx = self.new_act_idx






    def epsilon_greedy_policy(self, curr_ep):


        possible_act_vals = self.action_val[self.new_y][self.new_x]
        actions = self.actions


        epsilon = float(1.0/curr_ep)                                                          # 입실론-greedy 가 GLIE
        greedy_prob = 1 - epsilon                                                           # GLIE : greedy imp 는 1 - 1/k, random imp(epsilon) 는 1/mk


        val_max_idx    = np.argmax(possible_act_vals)                                                    # 입실론-greedy improvement 적용.
        max_action     = actions[val_max_idx]                                                              # max_action 바로 뽑을수도, 그냥 random 으로 뽑아야 될 수도있다.


        chosen_action = np.random.choice([max_action, actions], p=[greedy_prob, epsilon])





        if chosen_action == max_action:
            #print("max_idx = ", curr_ep, greedy_prob)                                                    #  random 으로 greedy 가 될수도 있다.
            return val_max_idx
        else:
            random_idx = np.random.choice(range(len(actions)))
            #print("curr_ep = ", curr_ep, epsilon)
            return random_idx



def plot(EP, Stand, King):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    graph = ax.plot(EP, Stand, EP, King)
    ax.legend(graph, ['Stand', 'King'], loc=0)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("time_steps")

    axes = plt.gca()
    axes.set_xlim([0, 200])
    axes.set_ylim([0, 2000])

    plt.show()


def plot2(EP, S1, S2, S3):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    graph = ax.plot(EP, S1, EP, S2, EP, S3)
    ax.legend(graph, ['[0,3]', '[1,6]', '[3,5]'], loc=0)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("time_steps")

    axes = plt.gca()
    axes.set_xlim([0, 200])
    axes.set_ylim([0, 1000])

    plt.show()




problem = WindyGridWorld(start=[0,3], gamma=1, alpha=0.1, episodes=2000, Mode="Standard")
St_path, St_time_step, St_EP = problem.Windy_move()

problem2 = WindyGridWorld(start=[1,6], gamma=1, alpha=0.1, episodes=2000, Mode="Standard")
St1_path, St1_time_step, St1_EP = problem2.Windy_move()

problem2 = WindyGridWorld(start=[3,5], gamma=1, alpha=0.1, episodes=2000, Mode="Standard")
St2_path, St2_time_step, St2_EP = problem2.Windy_move()


plot2(St_EP, St_time_step, St1_time_step, St2_time_step)





problem3 = WindyGridWorld(start=[0,3], gamma=1, alpha=0.1, episodes=2000, Mode="King")
K3_path, K3_time_step, K3_EP = problem3.Windy_move()


problem4 = WindyGridWorld(start=[1,6], gamma=1, alpha=0.1, episodes=2000, Mode="King")
K4_path, K4_time_step, K4_EP = problem4.Windy_move()


problem5 = WindyGridWorld(start=[3,5], gamma=1, alpha=0.1, episodes=2000, Mode="King")
K5_path, K5_time_step, K5_EP = problem5.Windy_move()


plot2(K5_EP, K3_time_step, K4_time_step, K5_time_step)













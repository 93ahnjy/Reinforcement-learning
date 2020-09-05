import numpy as np
import matplotlib.pyplot as plt

class Random_walk:

    def __init__(self, gamma, alpha, episodes):
        self.pos = 3
        self.new_pos = 3
        self.values = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]
        self.reward = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        self.gamma = gamma
        self.alpha = alpha
        self.episode = episodes

        self.values_log = []





    def random_walk_TD(self):
        self.pos = 3
        self.values = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]
        values_log = []

        ################################################
        for i in range(self.episode):

            self.pos = 3
            self.new_pos = 3
            if i in [0, 1, 10, 100]:
                values_log.append(self.values[1:6])

            ################################################
            while 1:
                direct = np.random.choice(["left", "right"])

                if direct == "left":
                    self.new_pos = self.pos - 1
                    self.update_val_TD()
                    if self.new_pos == 0:
                        print("BAD")
                        break

                else:
                    self.new_pos = self.pos + 1
                    self.update_val_TD()
                    if self.new_pos == 6:
                        print("GOOD")
                        break

            ################################################

            print("END episode", i)
            print("\n")

        #

        ###############################################


        return  values_log






    def random_walk_MDP(self):
        self.pos = 3
        self.values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        values_log = []


        ################################################
        for i in range(self.episode):

            if i in [0, 1, 10, 100]:
                values_log.append(self.values[1:6])

            ################################################
            while 1:
                old_values = []
                for j in range(1,6):
                    old_values = self.values
                    self.update_val_MDP(j)

                if old_values == self.values:
                    break
            ################################################

            print("END episode", i)
            print("\n")
        ################################################

        return  values_log





    def random_walk_MC(self):
        self.pos = 3
        self.values = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]
        values_log = []

        ################################################
        for i in range(self.episode):

            self.pos = 3
            self.new_pos = 3
            if i in [0, 1, 10, 100]:
                values_log.append(self.values[1:6])

            visited_pos = []
            returns = 0
            ################################################
            while 1:
                direct = np.random.choice(["left", "right"])


                if direct == "left":
                    self.pos -= 1
                    if self.pos == 0:
                        print("BAD")
                        break
                    visited_pos.append(self.pos)

                else:
                    self.pos += 1
                    if self.pos == 6:
                        returns = 1
                        print("GOOD")
                        break
                    visited_pos.append(self.pos)

            ################################################
            uniq_visit = []
            for p in visited_pos:
                if p not in uniq_visit:
                    uniq_visit.append(p)


            print("visited_pos =", visited_pos, "reward =", returns)
            for pos in uniq_visit:
                self.update_val_MC(pos, returns)

            print("END episode", i)
            print("\n")
        ################################################

        return  values_log



    def update_val_TD(self):
        self.values[self.pos] += self.alpha*(self.reward[self.new_pos] + self.gamma*self.values[self.new_pos] - self.values[self.pos])
        self.pos = self.new_pos
        print("new_pos =", self.pos, "values =", np.round(np.array(self.values), 3))



    def update_val_MDP(self, pos):
        self.values[pos] = 0.5*(self.reward[pos+1] + self.gamma*self.values[pos+1]) + 0.5*(self.reward[pos-1] + self.gamma*self.values[pos-1])
        print("values =", np.round(np.array(self.values), 3))



    def update_val_MC(self, pos, returns):
        self.values[pos] += self.alpha*(returns - self.values[pos])
        print("values =", np.round(np.array(self.values), 3))




def plot(values_log):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    p= ["A", "B", "C", "D", "E"]
    graph = ax.plot(p, values_log[0], p, values_log[1], p, values_log[2], p, values_log[3], p, values_log[4])
    ax.legend(graph, ['0', '1', '10', '100', 'true'], loc=0)
    ax.set_ylabel("Estimated Value")
    ax.set_xlabel("State")


    plt.show()




prob = Random_walk(gamma=1, alpha=0.1, episodes=101)

MDP_result = prob.random_walk_MDP()
true = MDP_result[3]

TD_result = prob.random_walk_TD()
TD_result.append(true)
print("TD : ", np.round(np.array(TD_result), 3))

MC_result = prob.random_walk_MC()
MC_result.append(true)
print("MC :", np.round(np.array(TD_result), 3))


plot(TD_result)
plot(MC_result)
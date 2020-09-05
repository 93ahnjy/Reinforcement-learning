import numpy as np
import matplotlib.pyplot as plt
import random
from math import*



# Tiling 및 Tiling 내 tile 개수 설정.
totalTilings = 4
Tilings_numX = 9
Tilings_numY = 9
numTile_in_Tiling = Tilings_numX * Tilings_numY
total_Tiles = numTile_in_Tiling * totalTilings

# Tiling 의 크기.
Tilings_Width = 1.7
Tilings_Height = 1.4

# tile 의 크기.
tile_W = Tilings_Width / (Tilings_numX-1)
tile_H = Tilings_Height / (Tilings_numY-1)








def tile_code(x, y):

    x += 1.2
    y += 0.7
    tileIdx_in_Tilings = []

    for i in range(0, totalTilings):
        # asymmetric 하게 offset 을 설정하기 위해.
        # tile 의 width 가 offset 의 unit.
        newX = x + (tile_W * (1 / totalTilings)) * i
        newY = y + (tile_W * (1 / totalTilings)) * i

        # i번째 tiling 에서 어느 (x,y) 위치 타일로 떨어지는가?
        tile_posX = newX / tile_W
        tile_posY = newY / tile_H


        # 이 때, tiling 의 idx 가 넘어가도 tile 의 idx 는 초기화되지 않는다.
        tile_posX += ((numTile_in_Tiling) * i)

        # 올림으로 tile 의 좌표를 최종적으로 구한다.
        tile_posX = floor(tile_posX)
        tile_posY = floor(tile_posY)

        # 현재 state(pos, vel) 은
        # 1번째 tiling 에서의 idx 는 k1,
        # 2번째 tiling 에서의 idx 는 k2,
        # 3번째 tiling 에서의 idx 는 k3......
        # 8번째 tiling 에서의 idx 는 k8 이런식으로 표기.
        tileIdx_in_Tilings.append((8 * tile_posY) + tile_posX)

    return tileIdx_in_Tilings






class MountainCar:

    # 1
    def __init__(self):
        self.position = -0.6 + random.random() * 0.2
        self.velocity = 0

        self.reward = 0
        self.returns = 0


        self.features_pos      = []
        self.new_features_pos  = []
        self.update_param = np.full(total_Tiles * 3,  0, dtype=float)
        self.goal = 0


    def reset(self):
        self.position = -0.6 + random.random() * 0.2
        self.velocity = 0

        self.reward = 0
        self.returns = 0
        self.features_pos = []
        self.goal = 0



    def Mountain_move(self):


        alpha = 0.5 / totalTilings
        gamma = 1
        episodes = 1000

        for i in range(episodes):
            self.reset()
            delta_q = np.full(total_Tiles * 3, 0, dtype=float)
            step = 0

            while 1:

                # tile coding 을 통해 feature vector 를 얻자.
                self.features_pos = tile_code(self.position, self.velocity)


                # 현재 state 인 position, velocity 는 정해진 상황.
                # Q(s,a)에서 action 별 value 를 찾자.
                action_value = [0, 0, 0]
                for feature_pos in self.features_pos:
                    action_value[0] += self.update_param[feature_pos]                     # backward
                    action_value[1] += self.update_param[feature_pos + total_Tiles]       # zero
                    action_value[2] += self.update_param[feature_pos + (2 * total_Tiles)] # forward

                # action 선택. self.action_value 도 관련있음.
                action = self.epsilon_greedy_policy(i+1, action_value)



                # 't+1' state 중 velocity 값 구하기.
                # 't' state 는 일단 유지.
                new_velocity = self.velocity + 0.001 * (action - 1) - 0.0025 * cos(3 * self.position)
                if self.velocity < -0.07:
                    new_velocity = -0.07
                elif self.velocity >= 0.07:
                    new_velocity = 0.06999999


                # 't+1' state 중 position 값 구하기.
                # 't' state 는 일단 유지.
                # bound 조건에 따라 reward 및 loop 통과.
                new_position = self.position + new_velocity
                if self.position >= 0.5:
                    self.goal = 1
                    self.reward = 1
                elif self.position < -1.2:
                    new_position = -1.2
                    new_velocity = 0.0
                    self.reward = -1.5
                elif action == 2:
                    self.reward = -1



                self.returns += self.reward



                for feature_pos in self.features_pos:
                    delta_q[feature_pos + (action * total_Tiles)] = 1


                if self.goal:
                    self.update_param += (alpha * (self.reward - action_value[action]) * delta_q)
                    break



                # update using Action value approximation
                self.new_features_pos = tile_code(new_position, new_velocity)

                # 현재 state 인 position, velocity 는 정해진 상황.
                # Q(s,a)에서 action 별 value 를 찾자.
                new_action_value = [0, 0, 0]
                for feature_pos in self.new_features_pos:
                    new_action_value[0] += self.update_param[feature_pos]  # backward
                    new_action_value[1] += self.update_param[feature_pos + total_Tiles]  # zero
                    new_action_value[2] += self.update_param[feature_pos + (2 * total_Tiles)]  # forward



                # action 선택. self.action_value 도 관련있음.
                new_action = self.epsilon_greedy_policy(i+1, new_action_value)
                self.update_param += alpha * (self.reward + gamma* new_action_value[new_action] - action_value[action]) * delta_q
                delta_q *= 0.9



                # update 끝났으니, new state 를 현재 state 로 변경.
                self.position = new_position
                self.velocity = new_velocity
                step += 1

                print("pos :", round(self.position, 5), " vel :", round(self.velocity, 5))
                print("action_values : ", action_value)
                print("step: ", step,  "ep : ", i, "\n")

            ################################################

            print("END episode", i)
            print("\n")

        ################################################




    def epsilon_greedy_policy(self, curr_ep, action_value):

        epsilon = float(1.0/curr_ep)                                                          # 입실론-greedy 가 GLIE
        greedy_prob = 1 - epsilon                                                           # GLIE : greedy imp 는 1 - 1/k, random imp(epsilon) 는 1/mk


        # 계산에 필요한 action 은 -1, 0, 1
        # 마침 action_idx 도 0, 1, 2 로 1씩밖에 차이 안남.
        if random.random() <= greedy_prob:
            return np.argmax(action_value)
        else:
            return np.random.randint(3)




a = MountainCar()
a.Mountain_move()
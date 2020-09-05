import numpy as np
from math import*

def init():
    position = -0.6 + np.random.random()*0.2
    return position, 0.0

def move(State,Action):
    position,velocity = State

    Reward = -1 if Action==1 else -1.5
    Action -= 1
    velocity += 0.001*Action - 0.0025*cos(3*position)
    if velocity < -0.07:
        velocity = -0.07
    elif velocity >= 0.07:
        velocity = 0.06999999
    position += velocity
    if position >= 0.5:
        return Reward, None

    if position < -1.2:
        position = -1.2
        velocity = 0.0
    return Reward,(position,velocity)
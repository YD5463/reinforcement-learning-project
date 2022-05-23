from collections import deque
from gym import spaces
import numpy as np
import gym


class ConcatObs(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.observation_space.shape[0], k), dtype=np.float32)
        self.original_frame = None

    def reset(self):
        ob = self.env.reset()
        self.original_frame = ob
        for _ in range(self.k):
            self.frames.append(ob)
        return self.get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.original_frame = ob
        self.frames.append(ob)
        return self.get_ob(), reward, done, info

    def get_ob(self):
        return np.stack(self.frames, axis=-1)

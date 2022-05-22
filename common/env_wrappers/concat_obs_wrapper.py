from collections import deque

from cv2 import cv2
from gym import spaces
import numpy as np
import gym


class ConcatObs(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = spaces.Box(low=0, high=1, shape=(74, 110, k), dtype=np.float32)

    @staticmethod
    def _preprocess(frame):
        processed = ConcatObs._rgb2gray(frame / 255)  # gray scale
        resized = cv2.resize(processed, (110, 84), interpolation=cv2.INTER_LINEAR)
        result = resized[10:97]
        return result.astype(np.float32)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(self._preprocess(ob))
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(self._preprocess(ob))
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.stack(self.frames, axis=-1)

    @staticmethod
    def _rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

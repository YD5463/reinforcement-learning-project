import math
from collections import deque

from cv2 import cv2
from gym import spaces
import numpy as np
import gym


class ConcatObs(gym.Wrapper):
    def __init__(self, env, k, resize: bool = True, skip_step: int = 2):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.resize = resize
        self.skip_step = skip_step
        shape = (74, 110, math.ceil(k / skip_step))
        if not resize:
            shape = (210, 160, math.ceil(k / skip_step))
        self.observation_space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        self.original_frame = None

    def preprocess(self, frame):
        processed = self._rgb2gray(frame / 255)  # gray scale
        if self.resize:
            processed = cv2.resize(processed, (110, 84), interpolation=cv2.INTER_LINEAR)
            processed = processed[10:97]
        return processed.astype(np.float32)

    def reset(self):
        ob = self.env.reset()
        self.original_frame = ob
        for _ in range(self.k):
            self.frames.append(self.preprocess(ob))
        return self.get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.original_frame = ob
        self.frames.append(self.preprocess(ob))
        return self.get_ob(), reward, done, info

    def get_ob(self):
        return np.stack([self.frames[i] for i in range(0, len(self.frames), self.skip_step)], axis=-1)

    @staticmethod
    def _rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

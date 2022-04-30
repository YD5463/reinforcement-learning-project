import datetime
import os
from pathlib import Path
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import gym
from gym import Env
from matplotlib import pyplot as plt
from common.env_wrappers.concat_obs_wrapper import ConcatObs

GAME_NAME = "SpaceInvaders-v0"


def label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text((im.size[0] / 20, im.size[1] / 18), f'Episode: {episode_num + 1}', fill=text_color)
    return im


def save_random_agent_gif(frames: list, dest_dir="outputs/videos/"):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(os.path.join(dest_dir, f"agent-{datetime.datetime.now()}.gif"), frames, fps=60)


def get_env(seed=42) -> Env:
    env = gym.make(GAME_NAME)
    env = ConcatObs(env, k=4)
    print(f"game name: {GAME_NAME}\nobservation space: {env.observation_space.shape}, "
          f"action space: {env.action_space.n}")
    env.seed(seed)
    return env


def run_dummy_demo(env: Env, steps: int, print_period: int):
    observation = env.reset()
    for i in range(steps):
        if i % print_period == 0:
            plt.imshow(observation)
            plt.show()
            observation, _, _, _ = env.step(1)


def gif_model_demo(predict_func, steps_num: int):
    env = get_env()
    state = np.array(env.reset())
    frames = []
    for i in range(steps_num):
        state, reward, done, _ = env.step(predict_func(state))
        state = np.array(state)
        frames.append(label_with_episode_number(state, i))
        if done:
            break
    save_random_agent_gif(frames)


def get_action_space_len(env: Env) -> int:
    return env.action_space.n

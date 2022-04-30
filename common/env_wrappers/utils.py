import uuid
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import gym
from gym import Env
from matplotlib import pyplot as plt
from common.env_wrappers.concat_obs_wrapper import ConcatObs

GAME_NAME = "SpaceInvaders-v0"


def save_agent_game(frames: List[np.ndarray], dest_dir="outputs/videos/"):
    dest_dir = Path(dest_dir) / f"agent-{str(uuid.uuid4())[:8]}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"frames number: {len(frames)}")
    for i,frame in enumerate(frames):
        im = Image.fromarray((frame*255).astype(np.uint8))
        im.save(dest_dir / f"{i}.jpeg")


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
        frames.append(state[..., 0])
        if done:
            break
    save_agent_game(frames)


def get_action_space_len(env: Env) -> int:
    return env.action_space.n

import os
import uuid
from collections import deque
from pathlib import Path
from typing import List
import cv2
import numpy as np
from PIL import Image
import gym
from gym import Env
from matplotlib import pyplot as plt
from common.env_wrappers.concat_obs_wrapper import ConcatObs

GAME_NAME = "SpaceInvaders-v0"
SAVED_FRAMES = 4


def get_env(seed=42) -> Env:
    env = gym.make(GAME_NAME)
    env = ConcatObs(env, k=SAVED_FRAMES)
    print(f"game name: {GAME_NAME}\nobservation space: {env.observation_space.shape}, "
          f"action space: {get_action_space_len(env)}")
    env.seed(seed)
    return env


def run_dummy_demo(env: Env, steps: int, print_period: int):
    observation = env.reset()
    for i in range(steps):
        if i % print_period == 0:
            plt.imshow(observation)
            plt.show()
            observation, _, _, _ = env.step(1)


# def save_agent_game(frames: List[np.ndarray], dest_dir="outputs/videos/"):
#     dest_dir = Path(dest_dir) / f"agent-{str(uuid.uuid4())[:8]}"
#     dest_dir.mkdir(parents=True, exist_ok=True)
#     print(f"frames number: {len(frames)}")
#     for i,frame in enumerate(frames):
#         im = Image.fromarray(frame.astype(np.uint8))
#         im.save(dest_dir / f"{i}.jpeg")


def save_agent_game_video(frames: List[np.ndarray], dest_dir="outputs/videos",filename=""):
    dest_path = f"{dest_dir}/{filename}.avi"
    print(f"frames number: {len(frames)}")
    size = (frames[0].shape[1],frames[0].shape[0])
    out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(*'XVID'), 30, size)
    for image in frames:
        out.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    out.release()


def gif_model_demo(predict_func, steps_num: int,prefix_name=""):
    env = gym.make(GAME_NAME)
    state = np.array(env.reset())
    real_frames = []
    frames = deque([], maxlen=SAVED_FRAMES)
    for i in range(steps_num):
        frames.append(ConcatObs.rgb2gray(state))
        if len(frames) < SAVED_FRAMES:
            continue
        state, reward, done, _ = env.step(predict_func(np.stack(frames, axis=-1)))
        real_frames.append(np.array(state).astype(np.uint8))
        if done:
            break
    save_agent_game_video(real_frames,filename=f"{prefix_name}-{str(uuid.uuid4())[:8]}")


def get_action_space_len(env: Env) -> int:
    return env.action_space.n

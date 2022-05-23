import uuid
from typing import List
import cv2
import numpy as np
import gym
from gym import Env
from common.env_wrappers.concat_obs_wrapper import ConcatObs

GAME_NAME = "SpaceInvaders-ram-v0"
SAVED_FRAMES = 24


def get_env(seed=42) -> ConcatObs:
    env = gym.make(GAME_NAME)
    env = ConcatObs(env, k=SAVED_FRAMES)
    print(f"game name: {GAME_NAME}\nobservation space: {env.observation_space.shape}, "
          f"action space: {get_action_space_len(env)}")
    env.seed(seed)
    return env


def run_dummy_demo(env: Env, steps: int, print_period: int):
    _ = env.reset()
    for i in range(steps):
        if i % print_period == 0:
            observation, _, _, _ = env.step(1)


def save_agent_game_video(frames: List[np.ndarray], dest_dir="outputs/videos", filename=""):
    dest_path = f"{dest_dir}/{filename}.avi"
    print(f"frames number: {len(frames)}")
    size = (frames[0].shape[1], frames[0].shape[0])
    out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(*'XVID'), 30, size)
    for image in frames:
        out.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    out.release()


def gif_model_demo(predict_func, steps_num: int, prefix_name=""):
    env = get_env()
    state = env.reset()
    real_frames = []
    for i in range(steps_num):
        state, reward, done, _ = env.step(predict_func(state))
        real_frames.append(env.original_frame)
        if done:
            break
    save_agent_game_video(real_frames, filename=f"{prefix_name}-{str(uuid.uuid4())[:8]}")


def get_action_space_len(env: Env) -> int:
    return env.action_space.n

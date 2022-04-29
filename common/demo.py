import os
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import gym


def label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text((im.size[0] / 20, im.size[1] / 18), f'Episode: {episode_num + 1}', fill=text_color)
    return im


def save_random_agent_gif(frames:list):
    dest_dir = "./videos/"
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(os.path.join('./videos/', 'random_agent.gif'), frames, fps=60)

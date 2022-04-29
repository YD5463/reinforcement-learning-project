import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import gym


def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text((im.size[0] / 20, im.size[1] / 18), f'Episode: {episode_num + 1}', fill=text_color)
    return im


def save_random_agent_gif(env):
    frames = []
    for i in range(500):
        state = env.reset()
        for t in range(500):
            action = env.action_space.sample()
            frame = env.render(mode='rgb_array')
            frames.append(_label_with_episode_number(frame, episode_num=i))
            state, _, done, _ = env.step(action)
            if done:
                print(f"breaking in {t}")
                break
    env.close()
    imageio.mimwrite(os.path.join('./videos/', 'random_agent.gif'), frames, fps=60)


env = gym.make('CartPole-v1')
save_random_agent_gif(env)

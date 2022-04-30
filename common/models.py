import time
from typing import Callable

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import Env
from tensorflow import keras
from keras import layers

from common.concat_obs_wrapper import ConcatObs

GAME_NAME = "SpaceInvaders-v0"


def get_env(seed=42) -> Env:
    env = gym.make(GAME_NAME)
    env = ConcatObs(env, k=4)
    print(f"game name: {GAME_NAME}\nobservation space: {env.observation_space.shape}, "
          f"action space: {env.action_space.n}")
    env.seed(seed)
    return env


def run_demo(env: Env, steps: int, print_period: int):
    observation = env.reset()
    for i in range(steps):
        if i % print_period == 0:
            plt.imshow(observation)
            plt.show()
            observation, _, _, _ = env.step(1)


def create_q_model(env: Env):
    inputs = layers.Input(shape=env.observation_space.shape)
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(env.action_space.n, activation="linear")(layer5)
    return keras.Model(inputs=inputs, outputs=action)


def train_with_main_and_target(
        env: Env, model_creator: Callable[[Env], keras.Model],
        gamma=0.99, epsilon=1.0, lr=0.00025,
        epsilon_min=0.1, epsilon_max=1.0,
        batch_size=32, update_target_network=10000, update_after_actions=4,
        max_memory_length=100000, epsilon_greedy_frames=1000000.0,
        max_reward=200, max_time_s=120*60,
):
    epsilon_interval = epsilon_max - epsilon_min
    model = model_creator(env)
    model_target = model_creator(env)

    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    action_history, state_history, state_next_history = [], [], []
    rewards_history, episode_reward_history = [], []
    done_history = []
    running_reward, episode_count, frame_count = 0, 0, 0
    num_actions = env.action_space.n
    loss_function = keras.losses.Huber()
    start_time = time.time()
    while (running_reward < max_reward) and (time.time() - start_time) < max_time_s:
        state = np.array(env.reset())
        episode_reward = 0

        for i in range(1, 10000):
            frame_count += 1
            if frame_count < 50000 or epsilon > np.random.rand(1)[0]:
                action = np.random.choice(env.action_space.n)
            else:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                action = tf.argmax(action_probs[0]).numpy()

            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)
            episode_reward += reward

            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                indices = np.random.choice(range(len(done_history)), size=batch_size)
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

                future_rewards = model_target.predict(state_next_sample)
                updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample
                masks = tf.one_hot(action_sample, num_actions)
                with tf.GradientTape() as tape:
                    q_values = model(state_sample)
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    loss = loss_function(updated_q_values, q_action)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if frame_count % update_target_network == 0:
                model_target.set_weights(model.get_weights())
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]
            if done:
                break
        print(f"done episode with reward {episode_reward}")
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        episode_count += 1


train_with_main_and_target(get_env(), create_q_model,max_time_s=60)

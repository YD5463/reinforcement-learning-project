import datetime
import time
from collections import deque
from typing import Callable
import numpy as np
import tensorflow as tf
from gym import Env
from tensorflow import keras
from common.env_wrappers.utils import get_action_space_len
from common.models import predict_action


class History:
    def __init__(self, max_memory_length, batch_size):
        self.batch_size = batch_size
        self.action_history = deque([], maxlen=max_memory_length)
        self.state_history = deque([], maxlen=max_memory_length)
        self.state_next_history = deque([], maxlen=max_memory_length)
        self.done_history = deque([], maxlen=max_memory_length)
        self.rewards_history = deque([], maxlen=max_memory_length)

    def append(self, action, state, state_next, done, reward):
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(done)
        self.rewards_history.append(reward)

    def __len__(self):
        return len(self.done_history)

    def get_sample(self):
        indices = np.random.choice(range(len(self)), size=self.batch_size)
        state_sample = np.array([self.state_history[i] for i in indices])
        state_next_sample = np.array([self.state_next_history[i] for i in indices])
        rewards_sample = [self.rewards_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]
        done_sample = tf.convert_to_tensor([float(self.done_history[i]) for i in indices])
        return state_sample, state_next_sample, rewards_sample, action_sample, done_sample


def decay_epsilon(epsilon):
    epsilon_max = 1.0
    epsilon_min = 0.1
    epsilon_interval = epsilon_max - epsilon_min
    epsilon_greedy_frames = 1000000.0
    epsilon -= epsilon_interval / epsilon_greedy_frames
    return max(epsilon, epsilon_min)


def main_and_target_train(
        env: Env, model_creator: Callable[[Env], keras.Model],
        gamma=0.99, epsilon=1.0, lr=0.00025,
        batch_size=32, update_target_network=10000, update_after_actions=4,
        max_memory_length=100000, max_time_s=120 * 60,
        num_first_exploration_steps=5000,checkpoint=5000,
) -> str:
    model = model_creator(env)
    model_target = model_creator(env)
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    history = History(max_memory_length, batch_size)
    running_reward, episode_count, frame_count = 0, 0, 0
    num_actions = get_action_space_len(env)
    loss_function = keras.losses.Huber()
    start_time = time.time()
    while (time.time() - start_time) < max_time_s:
        state = np.array(env.reset())
        episode_reward = 0
        done = False
        while not done:
            frame_count += 1
            if frame_count < num_first_exploration_steps or epsilon > np.random.rand(1)[0]:
                action = np.random.choice(num_actions)
            else:
                action = predict_action(model, state)
            epsilon = decay_epsilon(epsilon)

            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)
            episode_reward += reward
            history.append(action, state, state_next, done, reward)
            state = state_next

            if frame_count % update_after_actions == 0 and len(history) > batch_size:
                state_sample, state_next_sample, rewards_sample, action_sample, done_sample = history.get_sample()
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
                print(f"running reward: {running_reward} at episode {episode_count}, frame count {frame_count}")
        if episode_count % checkpoint == 0:
            save_output = f"outputs/model-checkpoint-{episode_count // checkpoint}"
            model.save(save_output)
        print(f"done episode with reward {episode_reward}")
        episode_count += 1
    save_output = f"outputs/model-{datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}"
    model.save(save_output)
    return save_output


def sarsa(
        env: Env, model_creator: Callable[[Env], keras.Model],
        gamma=0.99, epsilon=1.0, lr=0.00025, max_time_s=120 * 60,
        num_first_exploration_steps=5000,checkpoint=5000,
):
    model = model_creator(env)
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    running_reward, episode_count, frame_count = 0, 0, 0
    num_actions = get_action_space_len(env)
    loss_function = keras.losses.Huber()
    start_time = time.time()
    while (time.time() - start_time) < max_time_s:
        state = np.array(env.reset())
        episode_reward = 0
        done = False
        action = np.random.choice(num_actions)
        while not done:
            frame_count += 1
            epsilon = decay_epsilon(epsilon)

            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)
            episode_reward += reward

            with tf.GradientTape() as tape:
                if frame_count < num_first_exploration_steps or epsilon > np.random.rand(1)[0]:
                    action = np.random.choice(num_actions)
                else:
                    action = predict_action(model, state)
                updated_q_values = reward + gamma * action
                updated_q_values = updated_q_values * (1 - done) - done
                masks = tf.one_hot(action, num_actions)
                q_values = model(state)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action)
            state = state_next

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if episode_count % checkpoint == 0:
            save_output = f"outputs/model-checkpoint-{episode_count // checkpoint}"
            model.save(save_output)

        print(f"done episode with reward {episode_reward}")
        episode_count += 1
    save_output = f"outputs/model-{datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}"
    model.save(save_output)
    return save_output

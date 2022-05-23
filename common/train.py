import time
import uuid
from collections import deque
from typing import Callable
import numpy as np
import tensorflow as tf
from gym import Env
from matplotlib import pyplot as plt
from tensorflow import keras
from common.env_wrappers.utils import get_action_space_len
from common.models import predict_action


def unique_of(text: str) -> str:
    return f"{text}-{str(uuid.uuid4())[:8]}"


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


def q_learning_main_and_target_train(
        env: Env, model_creator: Callable[[Env], keras.Model],
        gamma=0.99, epsilon=1.0, lr=0.00025,
        batch_size=32, update_target_network=10000, update_after_actions=4,
        max_memory_length=100000, max_time_s=120 * 60,
        num_first_exploration_steps=5000, checkpoint=5000,
) -> str:
    _model_name = unique_of("q-learning-main-target")
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
            save_output = f"outputs/{_model_name}-checkpoint-{episode_count // checkpoint}"
            model.save(save_output)
        print(f"done episode with reward {episode_reward}")
        episode_count += 1
    save_output = f"outputs/{_model_name}"
    model.save(save_output)
    return save_output


def simple_sarsa(
        env: Env, model_creator: Callable[[Env], keras.Model],
        gamma: float = 0.99, epsilon: float = 1.0, lr: float = 0.00025,
        max_time_s: int = 120 * 60, num_first_exploration_steps: int = 5000,
        checkpoint: int = 5000,
):
    _model_name = unique_of("simple-sarsa")
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
                q_values = model(np.expand_dims(state, axis=0))
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action)
            state = state_next

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if episode_count % checkpoint == 0:
            save_output = f"outputs/{_model_name}-checkpoint-{episode_count // checkpoint}"
            model.save(save_output)

        print(f"done episode with reward {episode_reward}")
        episode_count += 1
    save_output = f"outputs/{_model_name}"
    model.save(save_output)
    return save_output


def actor_critic(
        env: Env, model_creator: Callable[[Env], keras.Model],
        gamma: float = 0.99, lr: float = 0.01, max_time_s: int = 120 * 60,
        checkpoint: int = 5000,
) -> str:
    saved_rewards = []
    _model_name = unique_of("actor-critic")
    model = model_creator(env)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    huber_loss = keras.losses.Huber()
    eps = np.finfo(np.float32).eps.item()
    episode_count, running_reward = 0, 0
    num_actions = get_action_space_len(env)
    start_time = time.time()
    while (time.time() - start_time) < max_time_s:
        state = env.reset()
        episode_reward = 0
        done = False
        action_probs_history, critic_value_history, rewards_history = [], [], []
        with tf.GradientTape() as tape:
            while not done:
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])

                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, action]))

                state, reward, done, _ = env.step(action)
                rewards_history.append(reward)
                episode_reward += reward

            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            actor_losses, critic_losses = [], []
            for log_prob, value, ret in zip(action_probs_history, critic_value_history, returns):
                diff = ret - value
                actor_losses.append(-log_prob * diff)
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_count += 1
        saved_rewards.append(running_reward)
        print(f"running reward: {running_reward} at episode {episode_count}")
        if episode_count % checkpoint == 0:
            save_output = f"outputs/{_model_name}-checkpoint-{episode_count // checkpoint}"
            model.save(save_output)

    save_output = f"outputs/{_model_name}"
    model.save(save_output)
    fig = plt.figure(figsize=(12, 12))
    plt.plot(saved_rewards)
    fig.suptitle('Episode Rewards', fontsize=20)
    plt.xlabel('episode number', fontsize=18)
    plt.ylabel('episode reward', fontsize=16)
    fig.savefig(f'outputs/{_model_name}-rewards.png', dpi=fig.dpi)
    return save_output


def reinforce_mc(
        env: Env, model_creator: Callable[[Env], keras.Model],
        gamma: float = 0.99, lr: float = 0.01, max_time_s: int = 120 * 60,
        checkpoint: int = 5000, max_history: int = 10000
):
    _model_name = unique_of("reinforce-mc")
    model = model_creator(env)
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=lr))
    states, actions, rewards = deque([], maxlen=max_history), deque([], maxlen=max_history), deque([],
                                                                                                   maxlen=max_history)
    scores, episodes = [], []
    action_size = get_action_space_len(env)
    state_shape = env.observation_space.shape
    start_time = time.time()
    i = 0
    while (time.time() - start_time) < max_time_s:
        state = env.reset()
        done = False
        score = 0
        while not done:
            policy = model.predict(tf.expand_dims(tf.convert_to_tensor(state), 0), batch_size=1).flatten()
            action = np.random.choice(action_size, 1, p=policy)[0]
            next_state, reward, done, info = env.step(action)

            states.append(state)
            rewards.append(reward)
            actions.append(action)

            score += reward
            state = next_state

        episode_length = len(states)
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length,) + state_shape)
        advantages = np.zeros((episode_length, action_size))

        for j in range(episode_length):
            update_inputs[j] = states[j]
            advantages[j][actions[j]] = discounted_rewards[j]

        model.fit(update_inputs, advantages, epochs=1, verbose=0)
        scores.append(score)
        print(f"episode: {i}, score:{score}")
        if i % checkpoint == 0:
            save_output = f"outputs/{_model_name}-checkpoint-{i // checkpoint}"
            model.save(save_output)
        i += 1

    save_output = f"outputs/{_model_name}"
    model.save(save_output)
    return save_output

import numpy as np
import tensorflow as tf
from gym import Env
from tensorflow import keras
from keras import layers
from common.env_wrappers.utils import get_action_space_len


def create_my_model1(env: Env):
    inputs = layers.Input(shape=env.observation_space.shape)
    layer6 = layers.Flatten()(inputs)
    layer7 = layers.Dense(64, activation="relu")(layer6)
    action = layers.Dense(get_action_space_len(env), activation="linear")(layer7)
    return keras.Model(inputs=inputs, outputs=action)


def create_actor_critic_model1(env: Env):
    inputs = layers.Input(shape=env.observation_space.shape)
    layer1 = layers.Flatten()(inputs)
    common = layers.Dense(64, activation="relu")(layer1)
    layer4 = layers.Dense(32, activation="relu")(common)
    action = layers.Dense(get_action_space_len(env), activation="softmax")(layer4)
    layer5 = layers.Dense(32, activation="relu")(common)
    critic = layers.Dense(1)(layer5)
    return keras.Model(inputs=inputs, outputs=[action, critic])


def reinforce_mc_model(env: Env):
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(get_action_space_len(env), activation='softmax'))
    return model


def predict_action(model, state: np.ndarray) -> np.ndarray:
    state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
    action_probs = model(state_tensor, training=False)
    return tf.argmax(action_probs[0]).numpy()


def predict_action_ac(model, state: np.ndarray) -> np.ndarray:
    state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
    action_probs, _ = model(state_tensor, training=False)
    return tf.argmax(np.squeeze(action_probs)).numpy()


def get_pre_trained_model(filepath: str):
    return lambda env: keras.models.load_model(filepath)

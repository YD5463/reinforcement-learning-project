import numpy as np
import tensorflow as tf
from gym import Env
from tensorflow import keras
from keras import layers
from common.env_wrappers.utils import get_action_space_len


def create_q_model(env: Env):
    inputs = layers.Input(shape=env.observation_space.shape)
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(get_action_space_len(env), activation="linear")(layer5)
    return keras.Model(inputs=inputs, outputs=action)


def create_complex_model(env: Env):
    inputs = layers.Input(shape=env.observation_space.shape)
    layer1 = layers.Conv2D(filters=10, kernel_size=(3, 3), strides=2, activation="relu")(inputs)
    layer2 = layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu")(layer1)
    layer3 = layers.MaxPooling2D(pool_size=(3, 3), padding='valid')(layer2)
    layer4 = layers.Conv2D(filters=5, kernel_size=(3, 3), strides=1, activation="relu")(layer3)
    layer5 = layers.Flatten()(layer4)
    layer6 = layers.Dense(512, activation="relu")(layer5)
    action = layers.Dense(get_action_space_len(env), activation="linear")(layer6)
    return keras.Model(inputs=inputs, outputs=action)


def predict_action(model, state: np.ndarray) -> np.ndarray:
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    return tf.argmax(action_probs[0]).numpy()

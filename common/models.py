import numpy as np
import tensorflow as tf
from gym import Env
from tensorflow import keras
from keras import layers
from common.env_wrappers.utils import get_action_space_len


def create_from_doc_model(env: Env):
    inputs = layers.Input(shape=env.observation_space.shape)
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(get_action_space_len(env), activation="linear")(layer5)
    return keras.Model(inputs=inputs, outputs=action)


def create_my_model1(env: Env):
    inputs = layers.Input(shape=env.observation_space.shape)
    layer1 = layers.Conv2D(filters=10, kernel_size=(3, 3), strides=2, activation="relu")(inputs)
    layer2 = layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu")(layer1)
    layer3 = layers.MaxPooling2D(pool_size=(3, 3), padding='valid')(layer2)
    layer4 = layers.Conv2D(filters=5, kernel_size=(3, 3), strides=1, activation="relu")(layer3)
    layer5 = layers.MaxPooling2D(pool_size=(3, 3), padding='valid')(layer4)
    layer6 = layers.Flatten()(layer5)
    layer7 = layers.Dense(512, activation="relu")(layer6)
    action = layers.Dense(get_action_space_len(env), activation="linear")(layer7)
    return keras.Model(inputs=inputs, outputs=action)


def create_actor_critic_model1(env: Env):
    inputs = layers.Input(shape=env.observation_space.shape)
    layer1 = layers.Conv2D(filters=10, kernel_size=(3, 3), activation="relu")(inputs)
    layer_max = layers.MaxPooling2D(pool_size=(3, 3), padding='valid')(layer1)
    layer2 = layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu")(layer_max)
    common = layers.MaxPooling2D(pool_size=(3, 3), padding='valid')(layer2)
#    layer3 = layers.Dense(512, activation="relu")(layers.Flatten()(common))
    action = layers.Dense(get_action_space_len(env), activation="softmax")(layers.Flatten()(common))
#    layer4 = layers.Dense(256, activation="relu")(common)
#    layer5 = layers.Dense(50, activation="relu")(layer4)
    critic = layers.Dense(1)(layers.Flatten()(common))
    return keras.Model(inputs=inputs, outputs=[action, critic])


def reinforce_mc_model(env: Env):
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=5, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), padding='valid'))
    model.add(layers.Dense(255, input_shape=env.observation_space.shape, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
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

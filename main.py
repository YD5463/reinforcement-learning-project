import keras.models

from common.train import (
    q_learning_main_and_target_train,
    simple_sarsa,
    actor_critic,
    reinforce_mc
)
from common.env_wrappers.utils import get_env, gif_model_demo
from common.models import (
    create_my_model1,
    predict_action,
    create_actor_critic_model1,
    reinforce_mc_model,
    predict_action_ac,
)


def experiment1():
    saved_path = q_learning_main_and_target_train(
        get_env(), create_my_model1, gamma=0.99, epsilon=1.0, lr=0.00025,
        batch_size=32, update_target_network=10000, update_after_actions=4,
        max_memory_length=100000,
        num_first_exploration_steps=5000, checkpoint=5000,
        max_time_s=60 * 60 * 5
    )
    print(f"saved path: {saved_path}")
    model = keras.models.load_model(saved_path)
    gif_model_demo(lambda state: predict_action(model, state), steps_num=10000)


def experiment2():
    saved_path = simple_sarsa(
        get_env(), create_my_model1, max_time_s=60 * 60 * 5,
        gamma=0.99, epsilon=1.0, lr=0.00025,
        num_first_exploration_steps=5000, checkpoint=5000,
    )
    print(f"saved path: {saved_path}")
    model = keras.models.load_model(saved_path)
    gif_model_demo(lambda state: predict_action(model, state), steps_num=10000)


def experiment3():
    saved_path = actor_critic(
        get_env(), create_actor_critic_model1, max_time_s=60 * 60 * 5,
        gamma=0.99, lr=0.00025, checkpoint=5000,
    )
    print(f"saved path: {saved_path}")
    model = keras.models.load_model(saved_path)
    gif_model_demo(lambda state: predict_action_ac(model, state), steps_num=10000)


def experiment4():
    saved_path = reinforce_mc(
        get_env(), reinforce_mc_model, max_time_s=60 * 60 * 5,
        gamma=0.99, lr=0.00025, checkpoint=5000,
    )
    print(f"saved path: {saved_path}")
    model = keras.models.load_model(saved_path)
    gif_model_demo(lambda state: predict_action(model, state), steps_num=10000)


if __name__ == '__main__':
    # experiment1()
    # experiment2()
    # experiment3()
    experiment4()

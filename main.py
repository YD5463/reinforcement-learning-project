from glob import glob

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


def test_model(saved_path,predict_func,test_name):
    print(f"saved path: {saved_path}")
    for checkpoint in glob(f"{saved_path}-checkpoint*/"):
        checkpoint_number = checkpoint.split("-")[-1][:-1]
        model = keras.models.load_model(checkpoint)
        gif_model_demo(lambda state: predict_func(model, state), steps_num=10000,
                       prefix_name=f"{test_name}-checkpoint-{checkpoint_number}")
    model = keras.models.load_model(saved_path)
    gif_model_demo(lambda state: predict_func(model, state), steps_num=10000, prefix_name=test_name)


def experiment1():
    saved_path = q_learning_main_and_target_train(
        get_env(), create_my_model1, gamma=0.99, epsilon=1.0, lr=0.00025,
        batch_size=32, update_target_network=10000, update_after_actions=4,
        max_memory_length=100000,
        num_first_exploration_steps=5000, checkpoint=5000,
        max_time_s=60 * 60 * 5
    )
    test_model(saved_path,predict_action,"q_learning_main_and_target_train")


def experiment2():
    saved_path = simple_sarsa(
        get_env(), create_my_model1, max_time_s=60 * 60 * 5,
        gamma=0.99, epsilon=1.0, lr=0.00025,
        num_first_exploration_steps=5000, checkpoint=5000,
    )
    test_model(saved_path, predict_action, "simple_sarsa")


def experiment3():
    saved_path = actor_critic(
        get_env(), create_actor_critic_model1, max_time_s=60 * 2,
        gamma=0.99, lr=0.00025, checkpoint=1,
    )
    test_model(saved_path,predict_action_ac, "actor-critic")


def experiment4():
    saved_path = reinforce_mc(
        get_env(), reinforce_mc_model, max_time_s=60 * 60 * 5,
        gamma=0.99, lr=0.00025, checkpoint=5000,
    )
    test_model(saved_path, predict_action, "reinforce_mc")


if __name__ == '__main__':
    # experiment1()
    # experiment2()
    experiment3()
    # experiment4()

from common.train import (
    q_learning_main_and_target_train,
    simple_sarsa,
    actor_critic,
    reinforce_mc
)
from common.env_wrappers.utils import get_env
from common.models import (
    create_my_model1,
    create_actor_critic_model1,
    reinforce_mc_model,
)


# def test_model(saved_path, predict_func, test_name):
#     print(f"saved path: {saved_path}")
#     for checkpoint in glob(f"{saved_path}-checkpoint*/"):
#         checkpoint_number = checkpoint.split("-")[-1][:-1]
#         model = keras.models.load_model(checkpoint)
#         gif_model_demo(lambda state: predict_func(model, state), steps_num=10000,
#                        prefix_name=f"{test_name}-checkpoint-{checkpoint_number}")
#     model = keras.models.load_model(saved_path)
#     gif_model_demo(lambda state: predict_func(model, state), steps_num=10000, prefix_name=test_name)


def experiment1():
    q_learning_main_and_target_train(
        get_env(), create_my_model1, gamma=0.99, epsilon=1.0, lr=0.00025,
        batch_size=32, update_target_network=10000, update_after_actions=4,
        max_memory_length=100000,
        num_first_exploration_steps=5000, checkpoint=5000,
        max_time_s=60 * 60 * 5
    )


def experiment2():
    simple_sarsa(
        get_env(), create_my_model1, max_time_s=60 * 60 * 12,
        gamma=0.95, epsilon=0.99, lr=0.001,
        num_first_exploration_steps=500, checkpoint=100,
    )


def experiment3():
    actor_critic(
        get_env(), create_actor_critic_model1, max_time_s=60 * 60 * 8,
        gamma=0.95, lr=0.01, checkpoint=10,
    )


def experiment4():
    reinforce_mc(
        get_env(), reinforce_mc_model, max_time_s=60 * 60 * 10,
        gamma=0.95, lr=0.001, checkpoint=100,
    )


if __name__ == '__main__':
    # experiment1()
    # experiment2()
    experiment3()
    # experiment4()

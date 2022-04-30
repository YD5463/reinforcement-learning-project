import keras.models

from common.train import main_and_target_train
from common.env_wrappers.utils import get_env, gif_model_demo
from common.models import create_q_model, predict_action

if __name__ == '__main__':
    saved_path = main_and_target_train(get_env(), create_q_model, max_time_s=60 * 60 * 5, max_reward=10000000)
    print(f"saved path: {saved_path}")
    model = keras.models.load_model(saved_path)
    gif_model_demo(lambda state: predict_action(model, state), steps_num=10000)

# import numpy as np
# import gym
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from collections import deque, Counter
# from tensorflow.contrib.layers import flatten, conv2d, fully_connected
#
# num_episodes = 800
# batch_size = 48
# input_shape = (None, 88, 80, 1)
# learning_rate = 0.001
# X_shape = (None, 88, 80, 4)
# discount_factor = 0.97
# global_step = 0
# copy_steps = 100
# steps_train = 4
# start_steps = 2000
# epsilon = 0.5
# eps_min = 0.05
# eps_max = 1.0
# eps_decay_steps = 500000
#
#
# def preprocess_observation(obs):
#     img = obs[25:201:2, ::2]
#     img = img.mean(axis=2)
#     color = 0
#     img[img == color] = 0
#     img = (img - 128) / 128 - 1
#     return img.reshape(88, 80)
#
#
# env = gym.make("SpaceInvaders-v0")
# n_outputs = env.action_space.shape[0]
# print(n_outputs)
# print(env.action_space.to_jsonable())
# observation = env.reset()
#
# for i in range(22):
#     if i > 20:
#         plt.imshow(observation)
#         plt.show()
#         observation, _, _, _ = env.step(1)
#
# stack_size = 4
# stacked_frames = deque([np.zeros((88, 80), dtype=np.int) for i in range(stack_size)], maxlen=4)
#
#
# def stack_frames(stacked_frames, state, is_new_episode):
#     frame = preprocess_observation(state)
#     if is_new_episode:
#         # Clear our stacked_frames
#         stacked_frames = deque([np.zeros((88, 80), dtype=np.int) for i in range(stack_size)], maxlen=4)
#         # Because we’re in a new episode, copy the same frame 4x, apply elementwise maxima
#         maxframe = np.maximum(frame, frame)
#         stacked_frames.append(maxframe)
#         stacked_frames.append(maxframe)
#         stacked_frames.append(maxframe)
#         stacked_frames.append(maxframe)
#         stacked_state = np.stack(stacked_frames, axis=2)
#     else:
#         maxframe = np.maximum(stacked_frames[-1], frame)
#         stacked_frames.append(maxframe)
#         stacked_state = np.stack(stacked_frames, axis=2)
#     return stacked_state, stacked_frames
#
#
# def q_network(X, name_scope):
#     initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0)
#     with tf.compat.v1.variable_scope(name_scope) as scope:
#         layer_1 = conv2d(X, num_outputs=32, kernel_size=(8, 8), stride=4, padding='SAME',
#                          weights_initializer=initializer)
#         layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4, 4), stride=2, padding='SAME',
#                          weights_initializer=initializer)
#         layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3, 3), stride=1, padding='SAME',
#                          weights_initializer=initializer)
#         flat = flatten(layer_3)
#         fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
#         output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
#         vars = {v.name[len(scope.name):]: v for v in
#                 tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
#     return vars, output
#
#
# def epsilon_greedy(action, step):
#     p = np.random.random(1).squeeze()  # 1D entries returned using squeeze
#     epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)  # Decaying policy with more steps
#     if np.random.rand() < epsilon:
#         return np.random.randint(n_outputs)
#     else:
#         return action
#
#
# buffer_len = 20000
# exp_buffer = deque(maxlen=buffer_len)
#
#
# def sample_memories(batch_size):
#     perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
#     mem = np.array(exp_buffer)[perm_batch]
#     return mem[:, 0], mem[:, 1], mem[:, 2], mem[:, 3], mem[:, 4]
#
#
# mainQ, mainQ_outputs = q_network(X, 'mainQ')
# targetQ, targetQ_outputs = q_network(X, 'targetQ')
# copy_op = [tf.compat.v1.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
# copy_target_to_main = tf.group(*copy_op)
#
# y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
# loss = tf.reduce_mean(input_tensor=tf.square(y - Q_action))
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(loss)
# init = tf.compat.v1.global_variables_initializer()
# loss_summary = tf.compat.v1.summary.scalar('LOSS', loss)
# merge_summary = tf.compat.v1.summary.merge_all()
# file_writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())
#
# with tf.compat.v1.Session() as sess:
#     init.run()
#     # for each episode
#     history = []
#     obs = env.reset()
#     for i in range(num_episodes):
#         done = False
#         epoch = 0
#         episodic_reward = 0
#         actions_counter = Counter()
#         episodic_loss = []
#         # First step, preprocess + initialize stack
#         obs, stacked_frames = stack_frames(stacked_frames, obs, True)
#         # while the state is not the terminal state
#         while not done:
#             actions = mainQ_outputs.eval(feed_dict={X: [obs], in_training_mode: False})
#             action = np.argmax(actions, axis=-1)
#             actions_counter[str(action)] += 1
#             action = epsilon_greedy(action, global_step)
#             next_obs, reward, done, _ = env.step(action)
#             next_obs, stacked_frames = stack_frames(stacked_frames, next_obs, False)
#             exp_buffer.append([obs, action, next_obs, reward, done])
#             if global_step % steps_train == 0 and global_step > start_steps:
#                 o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)
#                 o_obs = [x for x in o_obs]
#                 o_next_obs = [x for x in o_next_obs]
#                 next_act = mainQ_outputs.eval(feed_dict={X: o_next_obs, in_training_mode: False})
#                 y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1 - o_done)
#                 mrg_summary = merge_summary.eval(feed_dict={X: o_obs, y: np.expand_dims(y_batch, axis=-1), X_action: o_act, in_training_mode: False})
#                 file_writer.add_summary(mrg_summary, global_step)
#                 train_loss, _ = sess.run([loss, training_op],
#                                          feed_dict={X: o_obs, y: np.expand_dims(y_batch, axis=-1), X_action: o_act, in_training_mode: True})
#                 episodic_loss.append(train_loss)
#             if (global_step + 1) % copy_steps == 0 and global_step > start_steps:
#                 copy_target_to_main.run()
#             obs = next_obs
#             epoch += 1
#             global_step += 1
#             episodic_reward += reward
#             next_obs = np.zeros(obs.shape)
#             exp_buffer.append([obs, action, next_obs, reward, done])
#             obs = env.reset()
#             obs, stacked_frames = stack_frames(stacked_frames, obs, True)
#             history.append(episodic_reward)
#             print('Epochs per episode:', epoch, 'Episode Reward:', episodic_reward, "Episode number:", len(history))

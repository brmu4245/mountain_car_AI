# import gym
# import random, math, numpy as np
# import matplotlib.pyplot as plt
# import collections
#
#
# """
#     In order to run this file, make sure to comment out the other file: q_learning
#     and then uncomment this file. The last attempt still has the q_learning results.
# """
# # try something
# """
#     These are useful print commands to demonstrate the inital state of the environment.
# """
# #
# # env = gym.make('MountainCar-v0')
# #
# # print("Action Space = " + str(env.action_space))
# # print("observation space = " + str(env.observation_space))
# # print("High end of observation space is " + str(env.observation_space.high))
# # print("low end of observation space is " + str(env.observation_space.low))
# # print("the goal of the car = " + str(env.goal_position))
# # env.reset()
# # print("the step values are [ position,   velocity],  reward, done?, info:\n            " + str(env.step(0)))
#
# """
#     Useful information for MountainCar-v0:
#     - Positions:
#         env.min_position = -1.2
#         env.max_position = 0.6     --> position range [-1.2, 0.6]
#         env.max_speed = 0.07
#         env.min_speed = -0.07      --> speed range [-0.07, 0.07]
#         env.goal_position = 0.5
#         env.goal_velocity = 0
#     - Random:
#         env.seed() = random value
#
#     - Useful information:
#         env._height(position)   --> how high up from the lowest point
#         env.state               --> Tuple [position, velocity]
#
#     - env.step(action):
#         position, velocity = self.state
#         velocity += (action - 1)*self.force + math.cos(3*position)*(-env.gravity)
#             - self.force = 0.001
#             - self.gravity = 0.0025
#             - Therefore:
#                 velocity = velocity + (action - 1) * 0.001 + math.cos(3*position)*-(0.0025)
#                 example:
#
#         returns numpy array: self.state(position, velocity), reward, done, {}
#
#     - Action_Space: Discrete(3)
#         0 = push left
#         1 = no push
#         2 = push right
#         (information from: github.com/openai/gym/wiki/MountainCar-V0)
#
#     - Starting State:
#         Random position between -0.6 to -0.4
#         velocity = 0
#
#     - done = True when 0.5 position is achieved or 200 iterations are reached
#     - observation = Box (2)
#         0 = position (min = -1.2, max = 0.6)
#         1 = velocity (min = -0.07, max = 0.07)
# """
#
#
# def main():
#     env = gym.make('MountainCar-v0')
#     num_of_episodes = 100000
#     action = 2
#     q = 0
#     q_values = collections.Counter()
#     average_reward_per_episode = []
#     episode_count = 0
#     car = Q_learner(env, vel_weight=5, pos_weight=1, alpha=0.5, discount=0.5, epsilon=0.05, q_values=q_values)
#     # decay = car.epsilon / (num_of_episodes * 0.5)
#     for i_episode in range(num_of_episodes):
#         done = False
#         total_rewards_for_each_episode = []
#         random_counter = 0
#         env.reset()
#         # if car.epsilon > 0:
#         #     car.epsilon = car.epsilon - decay
#         if i_episode > (num_of_episodes * 0.5):
#             car.epsilon = 0
#         total_reward = 0
#         car.counter_initial += 1
#         if car.counter_initial % 1000 == 0:
#             print("Episode: " + str(car.counter_initial))
#         t = 0
#         while not done:
#             t += 1
#             if i_episode >= (num_of_episodes - 5):
#                 env.render()
#             # car.get_q(env, action)            # CHANGED POSITION -- didn't change anything
#             if np.random.random() >= (1 - car.epsilon):
#                 random_counter += 1
#                 action = random.randint(0, 2)
#             else:
#                 action = car.choose_action(env)
#             # car.get_q(env, action)            # NEXT CHANGE ...
#             state, reward, done, info = env.step(action)
#             car.get_q(env, action)
#             total_reward = total_reward + reward
#             if done:
#                 # print("Episode finished after {} timesteps".format(t + 1))
#                 break
#         episode_count = episode_count + 1
#         # print("The total reward was " + str(total_reward) + "! Random was visited " +
#         #       str(random_counter) + " times for episode "+ str(episode_count) + "." )
#
#         """
#             The following helps with printing out the results.
#         """
#         # total_reward = however many times up to 200 until the car makes it
#         total_rewards_for_each_episode.append(total_reward)  #list of all the rewards for each episode
#         average_reward = np.mean(total_rewards_for_each_episode)  # Average of the rewards for each episode
#         average_reward_per_episode.append(average_reward)  #list of averaged rewards for each episode
#
#     env.close()
#
#     print("List of rewards for all the trials in order: \n" + str(average_reward_per_episode))
#
#     plt.plot((np.arange(len(average_reward_per_episode)) + 1), average_reward_per_episode)
#     plt.xlabel('Episodes')
#     plt.ylabel('Average Reward')
#     plt.title('Approx-Q: Average Reward vs Episodes')
#     plt.savefig('rewards.pdf')
#     plt.close()
#
#     """
#         Main will call on an episode function that will calculate
#         the averages of each episode?
#     """
# class Q_learner():
#     def __init__(self, env, vel_weight, pos_weight, alpha, discount, epsilon, q_values):
#         self.env = env
#         self.vel_weight = vel_weight
#         self.pos_weight = pos_weight
#         self.alpha = alpha
#         self.discount = discount
#         self.epsilon = epsilon
#
#         self.counter_initial = 0
#         self.counter_post = 0
#         self.q_values = q_values
#
#     # def get_q(self, env, action, q):   # MISTAKE!!!!
#     def get_q(self, env, action):
#         """
#             Looking at the current state, we will update Q_value and weights.
#             The next 2 lines help to quantify the state space so that there are not millions of states.
#         """
#         velocity = float(f"{env.state[1]:.2f}")
#         position = float(f"{env.state[0]:.1f}")
#         state = (position, velocity)
#         feat_1_vel = env.state[1]
#         feat_2_pos = env.state[0]
#         """
#             These are print statements that help speed the testing process up instead of watching the car
#             which slows the process down.
#         """
#         if state[0] >= 0.5:
#             print("YYEEESSSSSSSSSSSSSSSSSSS!!!!!! WE MADE IT!!!!*************************************************************************")
#         elif state[0] > 0.3:
#             print("Wow! The car almost made it to the top!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#######################")
#         elif state[0] > 0:
#             print('THE CAR IS GETTING PAST 0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#
#
#         reward = self.calc_reward(env)
#         old_q = self.q_values[(state, action)]
#         new_q = (reward + self.discount * old_q)
#         difference = (new_q - old_q)
#         self.vel_weight = self.vel_weight + self.alpha * difference * feat_1_vel
#         self.pos_weight = self.pos_weight + self.alpha * difference * feat_2_pos
#         # print("Old q_value = " + str(self.q_values[(state, action)]))
#         self.q_values[(state, action)] = self.vel_weight * feat_1_vel + self.pos_weight * feat_2_pos
#
#     def calc_reward(self, env):
#         """
#             The following reward system showed some promise after 10,000 episodes...
#         """
#         # if env.state[1] >= 0.01 or env.state[1] <= -0.01:
#         #     reward = 900
#         # elif env.state[1] >= 0.008 or env.state[1] <= -0.008:
#         #     reward = 700
#         # elif env.state[1] >= 0.006 or env.state[1] <= -0.006:
#         #     reward = 500
#         # elif env.state[1] >= 0.004 or env.state[1] <= -0.004:
#         #     reward = 350
#         # elif env.state[1] >= 0.002 or env.state[1] <= -0.002:
#         #     reward = 20
#         # elif env.state[1] >= 0.001 or env.state[1] <= -0.001:
#         #     reward = 10
#         # else:
#         #     reward = -1000
#         # return reward
#
#         """
#                     The following is very random as far as success even with 100,000 episodes.
#                 """
#         if env.state[1] >= 0.05 or env.state[1] <= -0.05:
#             reward = 100
#         elif env.state[1] >= 0.03 or env.state[1] <= -0.03:
#             reward = 80
#         elif env.state[1] >= 0.01 or env.state[1] <= -0.01:
#             reward = 60
#         elif env.state[1] >= 0.007 or env.state[1] <= -0.007:
#             reward = 35
#         elif env.state[1] >= 0.004 or env.state[1] <= -0.004:
#             reward = 20
#         elif env.state[1] >= 0.001 or env.state[1] <= -0.001:
#             reward = 10
#         else:
#             reward = -60
#         return reward
#         """
#             The reward below seemed to produce better results, still sporadic in success.
#         """
#         # if (state[1] >= 0.001 or state[1] <= -0.001) and (state[0] > -0.4 or state[0] < -0.6):
#         #     reward = 6.0000005
#         # else:
#         #     reward = -1
#
#     def choose_action(self, env):
#         velocity = float(f"{env.state[1]:.2f}")  # * 100
#         position = float(f"{env.state[0]:.1f}")  # * 10
#         state = (position, velocity)
#         actions = [0, 1, 2]
#         best_action = 2
#         best_q = float("-inf")
#         for act in actions:
#             curr_q = self.q_values[(state, act)]
#
#             if curr_q > best_q:
#                 best_q = curr_q
#                 best_action = act
#         # print("For state" + str(state) + " and action " + str(best_action) + ",... the q-value = " + str(curr_q))
#         return best_action
#
#
# main()
# """
#     The following jibberish is the previous attempts at coding an algorithm without the use
#     of a class object. I found out that there are too many variables to keep track of
#     by coding this way.
# """
# # def choose_action(state, action, q_values, reward):
# #     q_values = update_q(state, action, q_values, reward)
# #     actions = [0,1,2]  # 0 = pull left, 1 = no push, 2 = push right
# #     best_value = float('-inf')
# #     rnd = random.randint(1,5)
# #     if rnd == 1:
# #         action = random.choice(actions)
# #     else:
# #         action = action_from_q_values(state, q_values)
# #
# #     # print("action = " + str(action))
# #     return action
# #
# #
# # def update_q(state, action, q_values, reward):
# #     actions = [0,1,2]  # 0 = pull left, 1 = no push, 2 = push right
# #     discount = 1
# #     alpha = 0.5
# #     # Need a high velocity...
# #     weight_1 = 10  # velocity
# #     feat_1_neg_vel = state[0]  #current velocity
# #     weight_2 = 5   # position (need 0.5)
# #     feat_2_pos_vel = state[1]
# #     q_state = w1 * f1 + w2 * f2
# #     if q_values[(state[1], action)] < q_state:
# #         q_diff = reward + discount * q_values[(state[1], action)] - q_state
# #         weight_1 = weight_1 + alpha * q_diff * state[0]
# #         weight_2 = weight_2 + alpha * q_diff * state[1]
# #         q_values[(state[1], action)] = q_state
# #
# #     return q_values
# #
# # def action_from_q_values(state, q_values):
# #     actions = [0,1,2]
# #     best_action = 1
# #     best_value = float('-inf')
# #     for action in actions:
# #         q_value = q_values[(state[1], action)]
# #         if q_value > best_value:
# #             best_value = q_value
# #             best_action = action
# #     return best_action
# #
# #
# #
# # def q(state, action):
# #     """
# #         This will determine the q_value given the state and action.
# #
# #         The value will be determined by using the position and velocity of the car.
# #
# #         Want low discount values for:
# #             - 0 to small pos velocity AND low position value -> action: pull
# #             - 0 to small neg velocity and high position value -> action: push
# #                 - if not 0, need to compare previous velocities to determine
# #                   if car is still moving forward or backwards for max momentum
# #                   or if the direction of velocity just changed
# #             -else: no push
# #                 - continuous neg velocity and mid-level position -> no push
# #                 - continuous pos velocity and mid-level position -> no push
# #
# #             - Therefore, while near an extreme position, the choices should be:
# #                 - if velocity = 0 AND previous velocity is (-) -> action: pull
# #                 - if velocity = 0 AND previous velocity is (+) -> action: push
# #                 - if velocity = (+) AND previous velocity is (-) -> action: pull
# #                 - if velocity = (-) AND previous velocity is (+) -> action: push
# #                 - else: no push
# #
# #         pos_vel, reward, done, info = env.step()
# #
# #         discount = 0.1
# #         reward = -1
# #         alpha = 0.5
# #         q_value = 0 (initially)
# #         current_sample = reward + discount * best_q_value
# #         current_sample = (-1) + 0.1 * (0) = -1   ...For state [(pos 0, vel 0) = -1]
# #         q_values[(state, action)] = (1 - alpha)* q_value + alpha * current_sample
# #         q_value for [(pos 0, vel 0)] = (0.5) * 0  + 0.5 * -1 = -0.5
# #         q_values[(state, action)] = q_values[(state, action)] + alpha * (sample - q_values[(state, action)])
# #
# #
# #
# #
# #     """
# #     pass
#
# # if using Model-Based Learning,
# # Each episode we will
# # for i_episode in range(20):
# #     observation = env.reset()
# #     for t in range(100):
# #         env.render()
# #         # print(observation)
# #         action = env.action_space.sample()
# #         observation, reward, done, info = env.step(action)
# #         print("observation = " + str(observation) + " reward = " + str(reward) + " info = " + str(info))
# #         if done:
# #             print("Episode finished after {} timesteps".format(t+1))
# #             break
#
#
# # env.close()
#

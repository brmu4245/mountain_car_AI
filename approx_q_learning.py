import gym
import random, math, numpy as np
import matplotlib.pyplot as plt
import collections

# try something
"""
    These are useful print commands to demonstrate the inital state of the environment.
"""
#
# env = gym.make('MountainCar-v0')
#
# print("Action Space = " + str(env.action_space))
# print("observation space = " + str(env.observation_space))
# print("High end of observation space is " + str(env.observation_space.high))
# print("low end of observation space is " + str(env.observation_space.low))
# print("the goal of the car = " + str(env.goal_position))
# env.reset()
# print("the step values are [ position,   velocity],  reward, done?, info:\n            " + str(env.step(0)))

"""
    Useful information for MountainCar-v0:
    - Positions:
        env.min_position = -1.2
        env.max_position = 0.6     --> position range [-1.2, 0.6]
        env.max_speed = 0.07
        env.min_speed = -0.07      --> speed range [-0.07, 0.07]
        env.goal_position = 0.5
        env.goal_velocity = 0
    - Random:
        env.seed() = random value

    - Useful information:
        env._height(position)   --> how high up from the lowest point
        env.state               --> Tuple [position, velocity]

    - env.step(action):
        position, velocity = self.state
        velocity += (action - 1)*self.force + math.cos(3*position)*(-env.gravity)
            - self.force = 0.001
            - self.gravity = 0.0025
            - Therefore:
                velocity = velocity + (action - 1) * 0.001 + math.cos(3*position)*-(0.0025)
                example:

        returns numpy array: self.state(position, velocity), reward, done, {}

    - Action_Space: Discrete(3)
        0 = push left
        1 = no push
        2 = push right
        (information from: github.com/openai/gym/wiki/MountainCar-V0)

    - Starting State:
        Random position between -0.6 to -0.4
        velocity = 0

    - done = True when 0.5 position is achieved or 200 iterations are reached
    - observation = Box (2)
        0 = position (min = -1.2, max = 0.6)
        1 = velocity (min = -0.07, max = 0.07)
"""


def main():
    env = gym.make('MountainCar-v0')
    num_of_episodes = 5000
    action = 1
    q = 0
    q_values = collections.Counter()
    average_reward_per_episode = []
    episode_count = 0
    car = Q_learner(env, vel_weight=2, pos_weight=1, alpha=1, discount=1, epsilon=0.1, q_values=q_values)
    decay = car.epsilon / num_of_episodes
    for i_episode in range(num_of_episodes):
        done = False
        total_rewards_for_each_episode = []
        random_counter = 0
        env.reset()
        if car.epsilon > 0:
            car.epsilon = car.epsilon - decay
        total_reward = 0
        t = 0
        while not done:
            t += 1
            if i_episode >= (num_of_episodes - 5):
                env.render()
            car.get_q(env, action, q)
            if np.random.random() >= (1 - car.epsilon):
                random_counter += 1
                action = random.randint(0, 2)
            else:
                action = car.choose_action(env, action, q)
            state, reward, done, info = env.step(action)
            total_reward = total_reward + reward
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        episode_count = episode_count + 1
        print("The total reward for round " + str(episode_count) + " is " + str(
            total_reward) + "! Random was visited " + str(random_counter) + " times.")
        # print("the initial q_values were visited " + str(car.counter_initial) + " times, and the other " + str(car.counter_post) + " times.")

        # total_reward = however many times up to 200 until the car makes it
        total_rewards_for_each_episode.append(total_reward)  #list of all the rewards for each episode
        average_reward = np.mean(total_rewards_for_each_episode)  # Average of the rewards for each episode
        average_reward_per_episode.append(average_reward)  #list of averaged rewards for each episode
        car.counter_initial = 0
        car.counter_post = 0

    env.close()

    print("List of rewards for all the trials in order: \n" + str(average_reward_per_episode))

    plt.plot((np.arange(len(average_reward_per_episode)) + 1), average_reward_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards.pdf')
    plt.close()

    """ 
        Main will call on an episode function that will calculate
        the averages of each episode?

    """


class Q_learner():
    def __init__(self, env, vel_weight, pos_weight, alpha, discount, epsilon, q_values):
        self.env = env
        self.vel_weight = vel_weight
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon

        self.counter_initial = 0
        self.counter_post = 0
        self.q_values = q_values

    def get_q(self, env, action, q):
        """
            Looking at the current state, we will update Q_value and weights.
        """
        velocity = float(f"{env.state[1]:.2f}")  # * 100
        position = float(f"{env.state[0]:.1f}")  # * 10
        state = (position, velocity)
        feat_1_vel = env.state[1]
        feat_2_pos = env.state[0]
        self.counter_post += 1
        # if state[1] >= 0.01 or state[1] <= -0.01:
        #     reward = 2000
        # elif state[1] >= 0.008 or state[1] <= -0.008:
        #     reward = 1000
        # elif state[1] >= 0.006 or state[1] <= -0.006:
        #     reward = 500
        # elif state[1] >= 0.004 or state[1] <= -0.004:
        #     reward = 350
        # elif state[1] >= 0.002 or state[1] <= -0.002:
        #     reward = 20
        # elif state[1] >= 0.001 or state[1] <= -0.001:
        #     reward = 10
        # else:
        #     reward = -100
        if (state[1] >= 0.01 or state[1] <= -0.01) and (state[0] > -0.5 or state[0] > -0.5):
            reward = 200
        else:
            reward = -100
        new_q = (reward + self.discount * q)
        difference = (new_q - q)
        self.vel_weight = self.vel_weight + self.alpha * difference * feat_1_vel
        self.pos_weight = self.pos_weight + self.alpha * difference * feat_2_pos
        # print("Old q_value = " + str(self.q_values[(state, action)]))
        self.q_values[(state, action)] = self.vel_weight * feat_1_vel + self.pos_weight * feat_2_pos

    def choose_action(self, env, action, q):
        velocity = float(f"{env.state[1]:.2f}")  # * 100
        position = float(f"{env.state[0]:.1f}")  # * 10
        state = (position, velocity)
        actions = [0, 1, 2]
        best_action = 1
        best_q = float("-inf")
        for act in actions:
            curr_q = self.q_values[(state, act)]

            if curr_q > best_q:
                best_q = curr_q
                best_action = act
        print("For state" + str(state) + " and action " + str(best_action) + ",... the q-value = " + str(curr_q))
        return best_action


main()
""" 
    The following jibberish is the previous attempts at coding an algorithm without the use
    of a class object. I found out that there are too many variables to keep track of 
    by coding this way.
"""
# def choose_action(state, action, q_values, reward):
#     q_values = update_q(state, action, q_values, reward)
#     actions = [0,1,2]  # 0 = pull left, 1 = no push, 2 = push right
#     best_value = float('-inf')
#     rnd = random.randint(1,5)
#     if rnd == 1:
#         action = random.choice(actions)
#     else:
#         action = action_from_q_values(state, q_values)
#
#     # print("action = " + str(action))
#     return action
#
#
# def update_q(state, action, q_values, reward):
#     actions = [0,1,2]  # 0 = pull left, 1 = no push, 2 = push right
#     discount = 1
#     alpha = 0.5
#     # Need a high velocity...
#     weight_1 = 10  # velocity
#     feat_1_neg_vel = state[0]  #current velocity
#     weight_2 = 5   # position (need 0.5)
#     feat_2_pos_vel = state[1]
#     q_state = w1 * f1 + w2 * f2
#     if q_values[(state[1], action)] < q_state:
#         q_diff = reward + discount * q_values[(state[1], action)] - q_state
#         weight_1 = weight_1 + alpha * q_diff * state[0]
#         weight_2 = weight_2 + alpha * q_diff * state[1]
#         q_values[(state[1], action)] = q_state
#
#     return q_values
#
# def action_from_q_values(state, q_values):
#     actions = [0,1,2]
#     best_action = 1
#     best_value = float('-inf')
#     for action in actions:
#         q_value = q_values[(state[1], action)]
#         if q_value > best_value:
#             best_value = q_value
#             best_action = action
#     return best_action
#
#
#
# def q(state, action):
#     """
#         This will determine the q_value given the state and action.
#
#         The value will be determined by using the position and velocity of the car.
#
#         Want low discount values for:
#             - 0 to small pos velocity AND low position value -> action: pull
#             - 0 to small neg velocity and high position value -> action: push
#                 - if not 0, need to compare previous velocities to determine
#                   if car is still moving forward or backwards for max momentum
#                   or if the direction of velocity just changed
#             -else: no push
#                 - continuous neg velocity and mid-level position -> no push
#                 - continuous pos velocity and mid-level position -> no push
#
#             - Therefore, while near an extreme position, the choices should be:
#                 - if velocity = 0 AND previous velocity is (-) -> action: pull
#                 - if velocity = 0 AND previous velocity is (+) -> action: push
#                 - if velocity = (+) AND previous velocity is (-) -> action: pull
#                 - if velocity = (-) AND previous velocity is (+) -> action: push
#                 - else: no push
#
#         pos_vel, reward, done, info = env.step()
#
#         discount = 0.1
#         reward = -1
#         alpha = 0.5
#         q_value = 0 (initially)
#         current_sample = reward + discount * best_q_value
#         current_sample = (-1) + 0.1 * (0) = -1   ...For state [(pos 0, vel 0) = -1]
#         q_values[(state, action)] = (1 - alpha)* q_value + alpha * current_sample
#         q_value for [(pos 0, vel 0)] = (0.5) * 0  + 0.5 * -1 = -0.5
#         q_values[(state, action)] = q_values[(state, action)] + alpha * (sample - q_values[(state, action)])
#
#
#
#
#     """
#     pass

# if using Model-Based Learning,
# Each episode we will
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         # print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print("observation = " + str(observation) + " reward = " + str(reward) + " info = " + str(info))
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break


# env.close()


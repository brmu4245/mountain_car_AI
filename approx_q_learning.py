import gym
import random, math, numpy as np
# import matplotlib.pyplot as plt
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
    num_of_actions_per_episode = 200
    action = 2
    reward = -1
    q = 0
    q_values = collections.Counter()
    episode_count = 0
    rewards_for_each_episode = []
    car = Q_learner(env, vel_weight=2, pos_weight=1, alpha=0.5, discount=0.9, epsilon=0.9, q_values=q_values)
    # state = env.reset()

    decay = car.epsilon / 1000
    for i_episode in range(num_of_episodes):
        random_counter = 0
        state = env.reset()
        if car.epsilon > 0:
            car.epsilon = car.epsilon - decay
        total_reward = 0
        for t in range(num_of_actions_per_episode):
            env.render()
            car.get_q(env, action, q)
            if np.random.random() >= (1 - car.epsilon):
                random_counter += 1
                action = random.randint(0, 2)
                car.get_q(env, action, q)
            else:
                action = car.choose_action(env, action, q)
            # print("position (goal = 0.5) and velocity = " + str(state))
            state, reward, done, info = env.step(action)
            total_reward = total_reward + reward
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        episode_count = episode_count + 1
        print("The total reward for round " + str(episode_count) + " is " + str(
            total_reward) + "! Random was visited " + str(random_counter) + " times.")
        # print("the initial q_values were visited " + str(car.counter_initial) + " times, and the other " + str(car.counter_post) + " times.")
        rewards_for_each_episode.append(total_reward)
        car.counter_initial = 0
        car.counter_post = 0

    env.close()

    print("List of rewards for all the trials in order: \n" + str(rewards_for_each_episode))

    # # Plot Rewards
    # plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    # plt.xlabel('Episodes')
    # plt.ylabel('Average Reward')
    # plt.title('Average Reward vs Episodes')
    # plt.savefig('rewards.jpg')
    # plt.close()

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
        # self.q_values = {}

    def get_q(self, env, action, q):
        """
            Looking at the current state, we will update Q_value and weights.
        """
        velocity = float(f"{env.state[1]:.2f}")  # * 100
        position = float(f"{env.state[0]:.1f}")  # * 10
        state = (position, velocity)
        feat_1_vel = env.state[1]
        feat_2_pos = env.state[0]
        if self.q_values[(state, action)] == 0 or self.q_values[(state, action)] == float("-inf"):
            q_value = self.vel_weight * feat_1_vel + feat_2_pos * self.pos_weight
            self.q_values[(state, action)] = q_value
            self.counter_initial += 1
        else:
            self.counter_post += 1
            # max_q = find_max_q(reward)
            # print("In get_q, the q_value = " + str(q_value))
            # print(str(self.vel_weight) + " * " + str(feat_1_vel) + " + " + str(feat_2_pos) + " * " + str(self.pos_weight))
            # q_value = float(f"{q_value:.2f}")
            #     difference = self.calc_diff(env, action, reward, q_value)
            #     print("velocity = " + str(state[1]))
            if state[1] >= 0.01 or state[1] <= -0.01:
                reward = 10
            elif state[1] >= 0.02 or state[1] <= -0.02:
                reward = 20
            elif state[1] >= 0.03 or state[1] <= -0.03:
                reward = 30
            elif state[1] >= 0.04 or state[1] <= -0.04:
                reward = 40
            elif state[1] >= 0.05 or state[1] <= -0.05:
                reward = 50
            elif state[1] >= 0.06 or state[1] <= -0.06:
                reward = 60
            else:
                reward = -30
            new_q = (reward + self.discount * q)
            difference = (new_q - q)
            self.vel_weight = self.vel_weight + self.alpha * difference * feat_1_vel
            self.pos_weight = self.pos_weight + self.alpha * difference * feat_2_pos
            # print("Old q_value = " + str(self.q_values[(state, action)]))
            self.q_values[(state, action)] = self.vel_weight * feat_1_vel + self.pos_weight * feat_2_pos
            # self.q_values[(state, action)] = self.q_values[(state, action)] + (self.alpha * difference)
            # print("New q_value = " + str(self.q_values[(state, action)]))

            # self.update_weights(difference, feat_1_vel, feat_2_pos)
            # print("new_q = " + str(new_q))
            # print("difference = " + str(difference))
            # print("feat_1_vel = " + str(feat_1_vel))
            # print("feat_2_pos = " + str(feat_2_pos))
            # print("For state = " + str(state) + ", the q_value = " + str(self.q_values[(state, action)]))
        # q_value = update_q(difference, env)
        # return q_value

    # def find_max_q(self, reward):
    #     for a in [0,1,2]:

    # max_q = reward + self.discount

    # def calc_diff(self, env, action, reward, q_value):
    #     """
    #         This is called from def get_q. It takes in the q_value from the current action
    #         and calculates the difference in order to update the weights.
    #
    #         This also quantifies the position and velocity space. There's like billions of
    #         configurations since each position and velocity can have 10+ decimal places. So,
    #         each time the dictionary of q_values are updated, I've quantified the
    #         position and velocity drastically reducing the number of states in the problem.
    #
    #     """
    #     velocity = float(f"{env.state[1]:.2f}")                 # These lines are the ones that
    #     position = float(f"{env.state[0]:.1f}")                 # help quantify the space.
    #     state = (position, velocity)
    #
    #     q_diff = 0
    #     q_list = []
    #     actions = [0,1,2]
    #     for act in actions:
    #         # print("for state: " + str(state) + " and action: " + str(act))
    #         if self.q_values[(state, act)] == 0:                    # In the Counter() dictionary, any key
    #             self.q_values[(state,act)] = float("-inf")          # not in the dictionary has an inital
    #         # print("Q = " + str(self.q_values[(state, act)]))        # value of 0. I'm setting this to -inf
    #         q_list.append(self.q_values[(state,act)])
    #         max_q = max(q_list)
    #         if max_q == float("-inf"):
    #             max_q = 0
    #         # print("max_q = " + str(max_q))
    #
    #     sample = reward + self.discount * max_q                     # SAMPLE is a calculation used to help
    #     # print("sample = " + str(reward) + " + " + str(self.discount) + " * " + str(max_q) + " = " + str(sample))                            # recalculate the weights of each feature.
    #     if self.q_values[(state, action)] == float("-inf"):
    #         q_diff = sample - (0)
    #         return q_diff
    #     else:
    #         q_diff = sample - self.q_values[(state, action)]
    #         # print("difference = " + str(q_diff))
    #         # if q_value > self.q_values[(state, act)]:
    #             # self.q_values[(state, act)] = q_value
    #             # sample = reward + self.discount * q_value
    #             # q_diff = sample - self.q_values[(state, act)]
    #             # self.vel_weight = self.vel_weight + self.alpha * q_diff * env.state[1]
    #             # self.pos_weight = self.pos_weight + self.alpha * q_diff * env.state[0]
    #         return q_diff

    # def update_weights(self, difference, feat_1_vel, feat_2_pos):
    # print("The following is the update_weights function: ")
    # print("vel_weight = " + str(self.vel_weight) + " + " + str(self.alpha) + " * " + str(difference) + " * " + str(feat_1_vel))
    # self.vel_weight = self.vel_weight + self.alpha * difference * feat_1_vel
    # self.pos_weight = self.pos_weight + self.alpha * difference * feat_2_pos

    # def difference(self, env, action, reward, q_value):
    #     velocity = float(f"{env.state[1]:.2f}") #* 100
    #     position = float(f"{env.state[0]:.1f}") #* 10
    #     q_diff = reward + self.discount * self.q_values[(position, velocity)] - q_value
    #     return q_diff

    def choose_action(self, env, action, q):
        velocity = float(f"{env.state[1]:.2f}")  # * 100
        position = float(f"{env.state[0]:.1f}")  # * 10
        state = (position, velocity)
        if self.q_values[(state, action)] == float("-inf"):  #
            # print("got here")
            velocity = float(f"{env.state[1]:.2f}")  # * 100
            position = float(f"{env.state[0]:.1f}")  # * 10
            state = (position, velocity)
            self.q_values[(state, action)] = self.get_q(env, action, q)
            new_action = random.randint(0, 2)
            # print("New-assigned random action = " + str(new_action))
            return new_action


        # self.epsilon = self.epsilon + self.epsilon
        # if self.epsilon % 1 == 0:
        #     self.rnd_num = self.rnd_num + 5
        # rnd_trial = random.randint(0,self.rnd_num)
        # if rnd_trial == 0:
        #     rnd_move = random.randint(0,2)
        #     print("EPSILON = " + str(self.epsilon))
        #     print("EPSILON!!! Exploring a new action!!!!")
        #     return rnd_move

        else:
            velocity = float(f"{env.state[1]:.2f}")  # * 100
            position = float(f"{env.state[0]:.1f}")  # * 10
            state = (position, velocity)
            actions = [0, 1, 2]
            best_action = 2
            best_q = float("-inf")
            for act in actions:
                curr_q = self.q_values[(state, act)]
                # print("For state" + str(state) + " and action " + str(act) + ",... the q-value = " + str(curr_q))
                if curr_q > best_q:
                    best_q = curr_q
                    best_action = act

            # print("Best action = " + str(best_action))
            return best_action

            # stored_q = self.q_values[(state, action)]
            # print("Old Q-value = " + str(stored_q) + " ==>> New Q-value = " + str(q_value))
            # print("for the following state: Position = " + str(position) + " and Velocity = " + str(velocity))
            # print("q_value (" + str(q_value) + ") > stored_q(" + str(stored_q) + ")")
            # q_for_all_a = []
            # for a in [0,1,2]:
            #     q = self.q_values[(state, a)]
            #     q_for_all_a.append(q)
            # if q_value > stored_q:
            #     self.q_values[(state, action)] = q_value
            #     best_action = action
            #     # print("New action assigned = " + str(best_action))
            # else:
            # best_action = self.get_action_from_q_values(env, state)     #__getattribute__(action)
        # return best_action

    # def get_action_from_q_values(self, env, state):
    #     actions = [0,1,2]
    #     best_act = 2
    #     best_q = float("-inf")
    #     for act in actions:
    #         curr_q = self.q_values[(state, act)]
    #         if curr_q > best_q:
    #             best_act = act
    #     # print("original action taken = " + str(best_act))
    #     return best_act

    # if self.q_values[(state[0], action)] == 0:         # if car is not moving at all...
    #     print("ACTION = PUSH RIGHT")
    #     return 2  # push right                         # push to the right.
    # if self.q_values[(state[0], action)] > 75:         # if car is going fast already,...
    #     print("ACTION = NO PUSH")
    #     return 1                                        # let it go no push ...
    # else:                                               # else, if car is going slow,...
    #     if state[1] < 0:                                # and heading left
    #         print("ACTION = PUSH LEFT")
    #         return 0                                    # push to the left
    #     if state[1] > 0:                                # or heading right,
    #         print("ACTION = PUSH RIGHT")
    #         return 2                                    # push to the right


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


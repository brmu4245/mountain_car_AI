import gym
import random, math, numpy as np
import matplotlib.pyplot as plt
import collections

""" 
    Results will be recorded in a pdf file titled 'rewards.pdf' in this directory of the program.
    Run this file, then open the pdf.
    The last results for 5000 episodes have been saved for your viewing!
    
    This file is ready to run. However, if you would like to run the 'approx_q_learning.py' file,
    then comment out this file and uncomment 'approx_q_learning.py' and run it.
"""


# try something
"""
    These are useful print commands to demonstrate the initial state of the environment.
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
    car = Q_learner(env, vel_weight=10, pos_weight=1, alpha=1, discount=1, epsilon=0.9, q_values=q_values)
    decay = car.epsilon / num_of_episodes
    for i_episode in range(num_of_episodes):
        done = False
        total_rewards_for_each_episode = []
        random_counter = 0
        env.reset()
        if car.epsilon > 0:                     # This is where I placed the epsilon feature...
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
                action = car.choose_action(env)
            state, reward, done, info = env.step(action)
            total_reward = total_reward + reward
            if done:
                # print("Episode finished after {} timesteps".format(t + 1))
                break
        episode_count = episode_count + 1
        if episode_count % 500 == 0:
            print("Episode: " + str(episode_count))
        # print("The total reward for round " + str(episode_count) + " is " + str(
        #     total_reward) + "! Random was visited " + str(random_counter) + " times.")


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
    plt.title('Q-Learning: Average Reward vs Episodes')
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
            Looking at the current state, we will update Q_value
            The state space is quantified in order to keep the dictionary values to a minimum.
        """
        velocity = float(f"{env.state[1]:.2f}")
        position = float(f"{env.state[0]:.1f}")
        state = (position, velocity)

        """
            These print statements helped speed up the process by not rendering the environment.
            It allowed me to see the progress of the car without watching each episode which could 
            get long when it comes to 100,000.
        """

        # if state[0] >= 0.5:
        #     print("YYEEESSSSSSSSSSSSSSSSSSS!!!!!! WE MADE IT!!!!*************************************************************************")
        # elif state[0] > 0.3:
        #     print("Wow! The car almost made it to the top!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#######################")
        # elif state[0] > 0:
        #     print('THE CAR IS GETTING PAST 0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        # Reward changes,... as long as the car is moving and not at the bottom of the hill, it gets a reward!
        if (state[1] > 0 or state[1] < 0) and (state[0] > -0.5 or state[0] < -0.5):
            reward = 10
        else:
            reward = -10
        # updated Q'(s,a) = alpha * reward + discount * Q(s,a)
        self.q_values[(state, action)] = self.alpha * reward + self.discount * self.q_values[(state, action)]

    def choose_action(self, env):
        """
            Quantifying the state space, then choosing the best action from the dictionary. This would be on-policy.
        """
        velocity = float(f"{env.state[1]:.2f}")
        position = float(f"{env.state[0]:.1f}")
        state = (position, velocity)
        actions = [0, 1, 2]
        best_action = 1
        best_q = float("-inf")
        # this loop is necessary as it extracts the action from the dictionary by looping through from the state and action.
        for act in actions:
            curr_q = self.q_values[(state, act)]
            if curr_q > best_q:
                best_q = curr_q
                best_action = act
        # print("For state" + str(state) + " and action " + str(best_action) + ",... the q-value = " + str(curr_q))
        return best_action


main()

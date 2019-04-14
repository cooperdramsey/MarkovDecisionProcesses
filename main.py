from gym import envs
import algorithms
import numpy as np
from timeit import default_timer as timer
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp

# Copyright 2019 Ang Peng Seng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def plot_epsilon_changes(lake, taxi, title, gamma, alpha):
    plt.figure()
    plt.title(title)
    plt.xlabel('Epsilon Value')
    plt.ylabel('Avg Moves To Goal')
    plt.grid()
    epsilon_scores = list(np.arange(0.1, 1, 0.1))
    # Do Q learning
    # train_times_1 = []
    # train_times_2 = []
    moves_1 = []
    moves_2 = []
    for e in epsilon_scores:
        # lake.reset()
        # taxi.reset()
        # _, time_1 = process_Q_taxi(taxi, alpha, gamma, e)
        # _, time_2 = process_Q_lake(lake, alpha, gamma, e)
        # train_times_1.append(time_1)
        # train_times_2.append(time_2)
        policy_taxi, _ = process_Q_taxi(taxi, alpha, gamma, e)
        policy_lake, _ = process_Q_lake(lake, alpha, gamma, e)
        moves_1.append(test_taxi(policy_taxi, is_q=True))
        moves_2.append(test_lake(policy_lake, is_q=True))

    plt.plot(epsilon_scores, moves_1, color="r", label='Taxi')
    plt.plot(epsilon_scores, moves_2, color="g", label='Lake')
    plt.legend()
    plt.show()


def plot_gamma_changes(lake, taxi, title, alpha, epsilon):
    plt.figure()
    plt.title(title)
    plt.xlabel('Gamma Value')
    plt.ylabel('Avg Moves To Goal')
    plt.grid()
    gamma_scores = list(np.arange(0.1, 1, 0.1))
    gamma_scores.append(0.95)
    # Do Q learning
    moves_1 = []
    moves_2 = []
    for g in gamma_scores:
        policy_taxi, _ = process_Q_taxi(taxi, alpha, g, epsilon)
        policy_lake, _ = process_Q_lake(lake, alpha, g, epsilon)
        moves_1.append(test_taxi(policy_taxi, is_q=True))
        moves_2.append(test_lake(policy_lake, is_q=True))
    plt.plot(gamma_scores, moves_1, color="r", label='Taxi')
    plt.plot(gamma_scores, moves_2, color="g", label='Lake')
    plt.legend()
    plt.show()


def plot_alpha_changes(lake, taxi, title, gamma, epsilon):
    plt.figure()
    plt.title(title)
    plt.xlabel('Alpha Value')
    plt.ylabel('Avg Moves To Goal')
    plt.grid()
    alpha_values = list(np.arange(0.1, 1, 0.1))
    # Do Q learning
    moves_1 = []
    moves_2 = []
    for a in alpha_values:
        policy_taxi, _ = process_Q_taxi(taxi, a, gamma, epsilon)
        policy_lake, _ = process_Q_lake(lake, a, gamma, epsilon)
        moves_1.append(test_taxi(policy_taxi, is_q=True))
        moves_2.append(test_lake(policy_lake, is_q=True))
    plt.plot(alpha_values, moves_1, color="r", label='Taxi')
    plt.plot(alpha_values, moves_2, color="g", label='Lake')
    plt.legend()
    plt.show()


def test_policy(policy, env, is_lake=False, is_q=False):
    if is_lake:
        goal = 1
    else:
        goal = 20
    if is_q:
        pol_counts = [algorithms.count(policy[0], env, goal) for i in range(10000)]
    else:
        pol_counts = [algorithms.count(policy, env, goal) for i in range(10000)]
    print("An agent using a policy which has been improved using policy-iterated takes about an average of " + str(
        int(np.mean(pol_counts)))
          + " steps to successfully complete its mission.")
    sns.distplot(pol_counts)
    plt.show()


def process_Q_lake(lake, _alpha, _gamma, _epsilon):
    # Q-Learning
    alpha = _alpha
    gamma = _gamma
    epsilon = _epsilon
    episodes = 100000
    start = timer()
    policy_lake_q = algorithms.Q_learning_train(lake, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
    end = timer()
    # print("Q Learning Lake: {}s".format(end - start))
    return policy_lake_q, end-start


def process_Q_taxi(taxi, _alpha, _gamma, _epsilon):
    # Q-Learning
    alpha = _alpha
    gamma = _gamma
    epsilon = _epsilon
    episodes = 100000
    start = timer()
    policy_taxi_q = algorithms.Q_learning_train(taxi, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
    end = timer()
    # print("Q Learning Taxi: {}s".format(end - start))
    return policy_taxi_q, end-start


def test_lake(policy, title='', is_q=False):
    env = envs.make('FrozenLake-v0')
    goal = 1.0
    if is_q:
        pol_counts = [algorithms.count(policy[0], env, goal) for i in range(10000)]
    else:
        pol_counts = [algorithms.count(policy, env, goal) for i in range(10000)]
    pol_counts = [x for x in pol_counts if x > -1]
    print("An agent using" + title + "takes about an average of " + str(
        int(np.mean(pol_counts)))
          + " steps to successfully complete its mission.")
    sns.distplot(pol_counts).set_title(title)
    plt.show()
    return np.mean(pol_counts)


def test_taxi(policy, title='', is_q=False):
    env = envs.make('Taxi-v2')
    goal = 20
    if is_q:
        pol_counts = [algorithms.count(policy[0], env, goal) for i in range(10000)]
    else:
        pol_counts = [algorithms.count(policy, env, goal) for i in range(10000)]
    pol_counts = [x for x in pol_counts if x > -1]
    print("An agent using" + title + "takes about an average of " + str(
        int(np.mean(pol_counts)))
          + " steps to successfully complete its mission.")
    sns.distplot(pol_counts).set_title(title)
    plt.show()
    return np.mean(pol_counts)


if __name__ == '__main__':
    lake = envs.make('FrozenLake-v0')
    print("Frozen Lake States: {}".format(lake.observation_space.n))

    taxi = envs.make('Taxi-v2')
    print("Taxi States: {}".format(taxi.observation_space.n))

    # Value iteration and policy iteration

    # # Value Iteration
    # start = timer()
    # policy_lake_v, V_lake, iters = algorithms.value_iteration(lake, discount_factor=0.99)
    # end = timer()
    # print("Value Iteration Lake: {}s in {} iters".format(end - start, iters))
    #
    # start = timer()
    # policy_taxi_v, V_taxi, iters = algorithms.value_iteration(taxi, discount_factor=0.99)
    # end = timer()
    # print("Value Iteration Taxi: {}s in {} iters".format(end - start, iters))
    #
    # lake.reset()
    # taxi.reset()
    #
    # # policy Iteration
    # random_policy_lake = np.ones([lake.env.nS, lake.env.nA]) / lake.env.nA
    # random_policy_taxi = np.ones([taxi.env.nS, taxi.env.nA]) / taxi.env.nA
    #
    # lake.reset()
    # taxi.reset()
    #
    # # Get policy Evals
    # policy_lake = algorithms.policy_eval(random_policy_lake, lake, discount_factor=0.95)
    # policy_taxi = algorithms.policy_eval(random_policy_taxi, taxi, discount_factor=0.95)
    #
    # # Random policy
    # start = timer()
    # policy_lake_p, lake_R_val, iters = algorithms.policy_iteration(lake, algorithms.policy_eval, discount_factor=0.99)
    # end = timer()
    # print("Policy Iteration Lake: {}s in {} iters".format(end - start, iters))
    #
    # start = timer()
    # policy_taxi_p, taxi_R_val, iters = algorithms.policy_iteration(taxi, algorithms.policy_eval, discount_factor=0.99)
    # end = timer()
    # print("Policy Iteration Taxi: {}s in {} iters".format(end - start, iters))
    #
    # lake.reset()
    # taxi.reset()
    #
    # # Check policy similatiry Lake
    # for x in range(len(policy_lake_p[0])):
    #     if not (policy_lake_p[0][x] == policy_lake_v[0][x]).all():
    #         print("Not the same Policy")
    #         break
    # print("Same Policy Lake")
    #
    # # Check policy similatiry Lake
    # for x in range(len(policy_taxi_p[0])):
    #     if not (policy_taxi_p[0][x] == policy_taxi_v[0][x]).all():
    #         print("Not the same Policy")
    #         break
    # print("Same Policy Taxi")

    # Q learning

    # plot_epsilon_changes(lake, taxi, 'Train Times at Different Epsilon')
    # plot_gamma_changes(lake, taxi, 'Avg Moves To Goal at Different Gamma')
    # plot_alpha_changes(lake, taxi, 'Avg Moves To Goal at Different Alpha')

    # p1 = mp.Process(target=plot_epsilon_changes, args=(lake, taxi, 'Avg Moves To Goal at Different Epsilon (a)', 0.3, 0.2,))
    # p2 = mp.Process(target=plot_epsilon_changes, args=(lake, taxi, 'Avg Moves To Goal at Different Epsilon (b)', 0.95, 0.2,))
    # p7 = mp.Process(target=plot_epsilon_changes, args=(lake, taxi, 'Avg Moves To Goal at Different Epsilon (c)', 0.3, 0.8,))
    # p8 = mp.Process(target=plot_epsilon_changes, args=(lake, taxi, 'Avg Moves To Goal at Different Epsilon (d)', 0.95, 0.8,))

    # p3 = mp.Process(target=plot_gamma_changes, args=(lake, taxi, 'Avg Moves To Goal at Different Gamma (a)', 0.8, 0.1,))
    # p4 = mp.Process(target=plot_gamma_changes, args=(lake, taxi, 'Avg Moves To Goal at Different Gamma (b)', 0.8, 0.9,))
    # p5 = mp.Process(target=plot_alpha_changes, args=(lake, taxi, 'Avg Moves To Goal at Different Alpha (a)', 0.95, 0.1,))
    # p6 = mp.Process(target=plot_alpha_changes, args=(lake, taxi, 'Avg Moves To Goal at Different Alpha (b)', 0.95, 0.9,))

    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    # p7.start()
    # p8.start()


    # policy_lake_q = process_Q_lake(lake, alpha, gamma, epsilon)
    # policy_taxi_q = process_Q_taxi(taxi, alpha, gamma, epsilon)

    # Test policies created by value iteration, policy iteration and Q-learning here
    # Create Graphs of tested policies
    # p1 = mp.Process(target=test_lake, args=(policy_lake_v, 'Value Iteration (Lake)',))
    # p2 = mp.Process(target=test_lake, args=(policy_lake_p, 'Policy Iteration (Lake)',))
    # p3 = mp.Process(target=test_lake, args=(policy_lake_q, 'Q Learning (Lake)', True))
    # p4 = mp.Process(target=test_taxi, args=(policy_taxi_v, 'Value Iteration (Taxi)',))
    # p5 = mp.Process(target=test_taxi, args=(policy_taxi_p, 'Policy Iteration (Taxi)',))
    # p6 = mp.Process(target=test_taxi, args=(policy_taxi_q, 'Q Learning (Taxi)', True))

    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()

    # Figure out average reward per move using the output Reward tables for value iteration, policy iteration and Q learning.
    # taxi_policy_v, taxi_R_val, _ = algorithms.policy_iteration(taxi, algorithms.policy_eval, discount_factor=0.99)
    # lake_policy_v, lake_R_val, _ = algorithms.policy_iteration(lake, algorithms.policy_eval, discount_factor=0.99)
    # Q-Learning LAKE
    alpha = 0.2
    gamma = 0.3
    epsilon = 0.9
    episodes = 100000
    start = timer()
    lake_policy, lake_Q_R = algorithms.Q_learning_train(lake, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
    end = timer()
    print('Lake:')
    print(end - start)

    # Q-Learning TAXI
    alpha = 0.2
    gamma = 0.95
    epsilon = 0.1
    episodes = 100000
    start = timer()
    taxi_policy, taxi_Q_R = algorithms.Q_learning_train(taxi, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
    end = timer()
    print('Taxi:')
    print(end - start)

    # test pararmters
    # test_lake(lake_policy_v, 'Policy Iteration Lake Dist. by Steps to Goal')
    # test_taxi(taxi_policy_v, 'Policy Iteration Taxi Dist. by Steps to Goal')
    # test_lake(lake_policy, 'Q-Learning Lake Dist. by Steps to Goal')
    # test_taxi(taxi_policy, 'Q-Learning Taxi Dist. by Steps to Goal')

    # Compute averages
    # v_lake = np.mean(lake_R_val)
    # v_taxi = np.mean(taxi_R_val)
    # q_lake = np.mean(lake_Q_R)
    # q_taxi = np.mean(taxi_Q_R)
    # print("Value Iteration Lake: {}".format(v_lake))
    # print("Value Iteration Taxi: {}".format(v_taxi))
    # print("Q Learning Lake: {}".format(q_lake))
    # print("Q Learning Taxi: {}".format(q_taxi))

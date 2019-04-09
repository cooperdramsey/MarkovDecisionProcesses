from gym import envs
import algorithms
import numpy as np
from timeit import default_timer as timer
import seaborn as sns


def test_policy(policy, env):
    pol_count = algorithms.count(policy[0], env)
    pol_counts = [algorithms.count(policy[0], env) for i in range(10000)]
    print("An agent using a policy which has been improved using policy-iterated takes about an average of " + str(
        int(np.mean(pol_counts)))
          + " steps to successfully complete its mission.")
    sns.distplot(pol_counts)


lake = envs.make('FrozenLake8x8-v0')
print("Frozen Lake States: {}".format(lake.observation_space.n))

taxi = envs.make('Taxi-v2')
print("Taxi States: {}".format(taxi.observation_space.n))

# Value Iteration
start = timer()
policy_lake_v, V_lake, iters = algorithms.value_iteration(lake, discount_factor=0.99)
end = timer()
print("Value Iteration Lake: {}s in {} iters".format(end - start, iters))

start = timer()
policy_taxi_v, V_taxi, iters = algorithms.value_iteration(taxi, discount_factor=0.99)
end = timer()
print("Value Iteration Taxi: {}s in {} iters".format(end - start, iters))

lake.reset()
taxi.reset()

# policy Iteration
random_policy_lake = np.ones([lake.env.nS, lake.env.nA]) / lake.env.nA
random_policy_taxi = np.ones([taxi.env.nS, taxi.env.nA]) / taxi.env.nA

lake.reset()
taxi.reset()

# Get policy Evals
policy_lake = algorithms.policy_eval(random_policy_lake, lake, discount_factor=0.95)
policy_taxi = algorithms.policy_eval(random_policy_taxi, taxi, discount_factor=0.95)

# Random policy
start = timer()
policy_lake_p, _, iters = algorithms.policy_iteration(lake, algorithms.policy_eval, discount_factor=0.99)
end = timer()
print("Policy Iteration Lake: {}s in {} iters".format(end - start, iters))


start = timer()
policy_taxi_p, _, iters = algorithms.policy_iteration(taxi, algorithms.policy_eval, discount_factor=0.99)
end = timer()
print("Policy Iteration Taxi: {}s in {} iters".format(end - start, iters))

lake.reset()
taxi.reset()

# Check policy similatiry Lake
for x in range(len(policy_lake_p[0])):
    if not (policy_lake_p[0][x] == policy_lake_v[0][x]).all():
        print("Not the same Policy")
        break
print("Same Policy Lake")


# Check policy similatiry Lake
for x in range(len(policy_taxi_p[0])):
    if not (policy_taxi_p[0][x] == policy_taxi_v[0][x]).all():
        print("Not the same Policy")
        break
print("Same Policy Taxi")

# Q-Learning
alpha = 0.2
gamma = 0.95
epsilon = 0.5
episodes = 100000
start = timer()
policy_lake_q = algorithms.Q_learning_train(lake, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
end = timer()
print("Q Learning Lake: {}s".format(end - start))

start = timer()
policy_taxi_q = algorithms.Q_learning_train(taxi, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
end = timer()
print("Q Learning Taxi: {}s".format(end - start))

# Test policies created by value iteration, policy iteration and Q-learning here

# Lake
# test_policy(policy_lake_v, lake)
# test_policy(policy_lake_p, lake)
# test_policy(policy_lake_q, lake)

# Taxi
test_policy(policy_taxi_v, taxi)
test_policy(policy_taxi_p, taxi)
test_policy(policy_taxi_q, taxi)

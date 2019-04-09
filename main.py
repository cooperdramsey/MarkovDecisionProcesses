import frozen_lake as fl
import blackjack as bj


env_1 = fl.FrozenLakeEnv()
print("Frozen Lake States: {}".format(env_1.observation_space.n))

env_2 = bj.BlackjackEnv()
print("Blackjack States: {}".format(env_2.n_states))


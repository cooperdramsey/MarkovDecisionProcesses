import numpy as np
import random
from IPython.display import clear_output


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.env.nS)
    while True:
        # TODO: Implement!
        delta = 0  # delta = change in value of state from one iteration to next

        for state in range(env.env.nS):  # for all states
            val = 0  # initiate value as 0

            for action, act_prob in enumerate(policy[state]):  # for all actions/action probabilities
                for prob, next_state, reward, done in env.env.P[state][
                    action]:  # transition probabilities,state,rewards of each action
                    val += act_prob * prob * (reward + discount_factor * V[next_state])  # eqn to calculate
            delta = max(delta, np.abs(val - V[state]))
            V[state] = val
        if delta < theta:  # break if the change in value is less than the threshold (theta)
            break
    return np.array(V)


def policy_iteration(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            for prob, next_state, reward, done in env.env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA
    iters = 0
    while True:
        iters += 1
        # Implement this!
        curr_pol_val = policy_eval_fn(policy, env, discount_factor)  # eval current policy
        policy_stable = True  # Check if policy did improve (Set it as True first)
        for state in range(env.env.nS):  # for each states
            chosen_act = np.argmax(policy[state])  # best action (Highest prob) under current policy
            act_values = one_step_lookahead(state, curr_pol_val)  # use one step lookahead to find action values
            best_act = np.argmax(act_values)  # find best action
            if chosen_act != best_act:
                policy_stable = False  # Greedily find best action
            policy[state] = np.eye(env.env.nA)[best_act]  # update
        if policy_stable:
            return policy, curr_pol_val, iters

    return policy, np.zeros(env.env.nS)


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.env.nA)
        for act in range(env.env.nA):
            for prob, next_state, reward, done in env.env.P[state][act]:
                A[act] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.env.nS)
    iters = 0
    while True:
        iters += 1
        delta = 0  # checker for improvements across states
        for state in range(env.env.nS):
            act_values = one_step_lookahead(state, V)  # lookahead one step
            best_act_value = np.max(act_values)  # get best action value
            delta = max(delta, np.abs(best_act_value - V[state]))  # find max delta across all states
            V[state] = best_act_value  # update value to best action value
        if delta < theta:  # if max improvement less than threshold
            break
    policy = np.zeros([env.env.nS, env.env.nA])
    for state in range(env.env.nS):  # for all states, create deterministic policy
        act_val = one_step_lookahead(state, V)
        best_action = np.argmax(act_val)
        policy[state][best_action] = 1

    # Implement!
    return policy, V, iters


def Q_learning_train(env, alpha, gamma, epsilon, episodes):
    """Q Learning Algorithm with epsilon greedy

    Args:
        env: Environment
        alpha: Learning Rate --> Extent to which our Q-values are being updated in every iteration.
        gamma: Discount Rate --> How much importance we want to give to future rewards
        epsilon: Probability of selecting random action instead of the 'optimal' action
        episodes: No. of episodes to train on

    Returns:
        Q-learning Trained policy

    """
    """Training the agent"""

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    # Initialize Q table of 500 x 6 size (500 states and 6 actions) with all zeroes
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(1, episodes + 1):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space randomly
            else:
                action = np.argmax(q_table[state])  # Exploit learned values by choosing optimal values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        # if i % 100 == 0:
        #     clear_output(wait=True)
        #     print(f"Episode: {i}")
    # Start with a random policy
    policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA

    for state in range(env.env.nS):  # for each states
        best_act = np.argmax(q_table[state])  # find best action
        policy[state] = np.eye(env.env.nA)[best_act]  # update

    # print("Training finished.\n")
    return policy, q_table


def count(policy, env):
    curr_state = env.reset()
    counter = 0
    reward = None
    while reward != 20:
        state, reward, done, info = env.step(np.argmax(policy[curr_state]))
        curr_state = state
        counter += 1
    return counter

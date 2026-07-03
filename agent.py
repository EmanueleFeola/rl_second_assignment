import numpy as np
import sys
from utils import roll_4_dice


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    nA = env.action_space.n

    Q = {}

    episode_rewards = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)

    def state_key(state):
        return tuple(state)

    def ensure_state(state):
        s = state_key(state)
        if s not in Q:
            Q[s] = np.zeros(nA)
        return s

    def epsilon_greedy_action(s):
        action_probs = np.ones(nA) * epsilon / nA
        best_action = np.argmax(Q[s])
        action_probs[best_action] += 1.0 - epsilon
        return np.random.choice(np.arange(nA), p=action_probs)

    for i in range(num_episodes):
        if (i + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i + 1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        s = ensure_state(state)

        action = epsilon_greedy_action(s)

        done = False
        t = 1

        while not done:
            next_state, reward, done, _ = env.step(action)
            s_next = ensure_state(next_state)

            vec_action = get_legal_moves(env.get_state())
            next_action = vec_action[0]

            episode_rewards[i] += reward
            episode_lengths[i] = t
            t += 1

            td_target = reward + discount_factor * Q[s_next][next_action]
            td_delta = td_target - Q[s][action]
            Q[s][action] += alpha * td_delta

            s = s_next
            action = next_action

    return Q, episode_rewards, episode_lengths


def sarsa_new(env, num_episodes=10000, alpha=0.1, epsilon=0.1, n=10, lamb=0.5):
    """
    n-SARSA or SARSA(lambda) algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        n: number of steps for n-step SARSA
        lamb: lambda value for SARSA(lambda)

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a nested dictionary mapping state -> action -> value.
        first_state_value: list with the value of Q in each episode for a chosen first state-action pair
    """

    # Initialise value function: nested dictionary that maps state -> action -> value.
    # Each key of Q should be a state, while the value is another dictionary
    # This nested dictionary should have actions as keys, and value function values as the correspdonding value of the key
    Q = {}

    """
    You can intialise the entire value function Q beforehand. This does require enumerating all states and actions in advance.
    Alternatively, you can add states to Q while running SARSA. Whenever you encounter a new state and/or action, you can add
    it to the dictionary.
    """

    first_state_values = []
    """
    Choose one of the first state-action pairs. The value of this state-action pair should be saved to the list
    first_state_values in every episode.
    """

    # Loop over the episodes
    for episode in range(num_episodes):
        # Print every 100 episodes.
        if (episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(episode + 1, num_episodes), end="")
            sys.stdout.flush()

        """
        Choose to either implement n-step SARSA or SARSA(lambda) here
        """

    return Q, first_state_values

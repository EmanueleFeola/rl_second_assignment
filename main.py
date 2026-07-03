from RoyalGameOfUr import RoyalGameOfUr
from agent import sarsa
import numpy as np
from utils import roll_4_dice

if __name__ == "__main__":
    n_pieces = 2
    env = RoyalGameOfUr(n_pieces)
    # env.render([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [1, 4, 8]) # this must throw error. 2 pieces in battle zone.
    # env.render([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [1, 2, 3, 4, 13, 14])  # full board
    # env.render([1, 2, 3, 4, 13, 14], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # full board

    # Q, episode_rewards, episode_lengths = sarsa(env, 1000)

    state = env.reset()
    done = False
    while not env.game_over:
        vec_a = env.get_valid_actions(flag_p1_turn=True)
        action = vec_a[0]  # trivial to test the game evolution
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render([state[0], state[1]], [state[2], state[3]])

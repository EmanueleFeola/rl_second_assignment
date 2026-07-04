from RoyalGameOfUr import RoyalGameOfUr
from agent import sarsa
import numpy as np
import os
from utils import load_Q, save_Q

filepath_q_function = "q_function_1000000.npz"


if __name__ == "__main__":
    n_pieces = 2  # not used inside env., i.e., only 2 pieces are considered to keep logic simple (as indicated in assignment description!)
    flag_verbose = False
    env = RoyalGameOfUr(n_pieces, flag_verbose)
    Q = None

    if os.path.exists(filepath_q_function):
        Q = load_Q(filepath_q_function)
        print(f"[main] Loaded precomputed Q from {filepath_q_function}")
    else:
        Q, _, _ = sarsa(env, 1000000)
        save_Q(Q, filepath_q_function)
        print(f"[main] Computed and saved Q to {filepath_q_function}")

    state = env.reset()
    while not env.game_over:
        vec_a = env.get_valid_actions(flag_p1_turn=True)
        action = np.argmax(Q[state])
        next_state, reward, _, _ = env.step(action)
        state = next_state
        env.render([state[0], state[1]], [state[2], state[3]])

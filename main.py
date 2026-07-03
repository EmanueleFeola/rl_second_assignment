from RoyalGameOfUr import RoyalGameOfUr
from agent import sarsa


if __name__ == "__main__":
    # Play the royal game of ur
    n_pieces = 2
    env = RoyalGameOfUr(n_pieces)
    # env.render([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [1, 4, 8]) # this must throw error. 2 pieces in battle zone.
    # env.render([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [1, 2, 3, 4, 13, 14])  # full board
    # env.render([1, 2, 3, 4, 13, 14], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # full board

    Q, episode_rewards, episode_lengths = sarsa(env, 10000)

    # state = env.reset()
    # done = False
    # while not env.game_over:
    #     action = np.argmax(Q[state])
    #     next_state, reward, done, _ = env.step(action)
    #     print("State:%s, action: %s, next state: %s" % (state, env.actions[action], next_state))
    #     state = next_state
    #     env.render([state[0], state[1]], [state[2], state[3]])

    # while not env.game_over:
    #     next_state, _ = env.step()
    #     env.render([env.x11, env.x12], [env.x21, env.x22])

    # if env.counter == 3:
    #     exit(0)

    #     # time.sleep(0.5)

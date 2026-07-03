import numpy as np
import gymnasium as gym
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import itertools
import random


def roll_4_dice():
    """
    roll 4 binary dices and return sum
    """

    sum = 0
    for _ in range(4):
        d_i = random.random()
        # print(f"[roll_4_dice] {d_i}")

        if d_i > 0.5:
            sum = sum + 1

    return sum


def append_move(vec_moves, move):
    """
    append move to vec_moves is move if not already included.
    return None because vec_moves is a mutable object
    """
    if move not in vec_moves:
        vec_moves.append(move)


def get_legal_moves(x11, x12, x21, x22, action):
    """
    Given current state (x11, x12, x21, x22) and action,
    return all legal next states for the active player.

    State convention:
        x11, x12 = active player's pieces
        x21, x22 = opponent's pieces

    Position convention:
        0  = start
        15 = final / completed
        1..4   = active player's private entry path
        5..12  = shared battle path
        13..14 = active player's private exit path

    Capture rule:
        Captures can only happen on shared squares 5..12.
        Square 8 is protected/starred, so landing on an opponent there is illegal.
    """
    # print(f"[get_legal_moves] action {action}")

    vec_moves = []

    x_final = RoyalGameOfUr.x_final
    shared_squares = range(5, 13)   # 5, ..., 12
    protected_shared_squares = [8]  # shared rosette/star square

    # If action is 0, no piece moves.
    if action == 0:
        return [[x11, x12, x21, x22]]

    vec_p1 = [x11, x12]

    for i in range(2):
        x_curr = vec_p1[i]
        x_other = vec_p1[1 - i]

        # Finished pieces cannot move.
        if x_curr == x_final:
            continue

        x_next = x_curr + action

        # Cannot move beyond final.
        if x_next > x_final:
            continue

        # Cannot land on own other piece, except at final.
        if x_next != x_final and x_next == x_other:
            continue

        # Initialize candidate next state.
        next_x11 = x11
        next_x12 = x12
        next_x21 = x21
        next_x22 = x22

        # Move active player's selected piece.
        if i == 0:
            next_x11 = x_next
        else:
            next_x12 = x_next

        # Reaching final is always legal.
        if x_next == x_final:
            append_move(vec_moves, [next_x11, next_x12, next_x21, next_x22])
            continue

        # Captures/collisions only matter on the shared path.
        if x_next in shared_squares:

            # Opponent piece 1 is on the same shared square.
            if x_next == x21:
                if x_next in protected_shared_squares:
                    # Cannot capture on protected shared square.
                    continue

                # Capture opponent piece 1.
                next_x21 = 0
                append_move(vec_moves, [next_x11, next_x12, next_x21, next_x22])
                continue

            # Opponent piece 2 is on the same shared square.
            if x_next == x22:
                if x_next in protected_shared_squares:
                    # Cannot capture on protected shared square.
                    continue

                # Capture opponent piece 2.
                next_x22 = 0
                append_move(vec_moves, [next_x11, next_x12, next_x21, next_x22])
                continue

        # Normal move.
        append_move(vec_moves, [next_x11, next_x12, next_x21, next_x22])

    # If no legal moves exist, stay in the same state.
    if len(vec_moves) == 0:
        append_move(vec_moves, [x11, x12, x21, x22])

    return vec_moves


class RoyalGameOfUr(gym.Env):
    x_final = 15
    vec_x_star = [4, 8, 14]

    def __init__(self, n_piece):
        # self.n_piece = n_piece  # Number of pieces per player
        # for now i assume we only have 2 pieces

        # define the states of the pieces s=(x11, ..., x22, dice_sum)
        self.x11 = None  # piece 1 of player 1
        self.x12 = None  # piece 2 of player 1
        self.x21 = None  # piece 1 of player 2
        self.x22 = None  # piece 2 of player 2
        self.dice_sum = None

        self.counter = None
        self.game_over = None

        self.reset()

    """
    The initialisation function is already given above, but you should still fill in the other elements of this function.
    The other functions that should be defined are
    
    1. The reset function, which should set the clear the board, set the number of scored pieces per player to zero, and start
    with the turn of player 1
    2. The step function, which gives the next state, reward, and done given the current state, roll, and action.
    (3.) It will be very useful to define a function that gives the possible moves for the current player given the board state
    and roll.
    (4.) It may also be useful to define a function that rolls the dice, but you can also make this part of the step function.
    
    A render function has been given which allows you to visualise episodes of the game being played.
    """

    def reset(self):
        # reset all pieces to zero, i.e., not on the board.
        self.x11 = 0
        self.x12 = 0
        self.x21 = 0
        self.x22 = 0
        self.game_over = False
        self.counter = 0

    def step_p1(self, action):
        vec_moves = get_legal_moves(self.x11, self.x12, self.x21, self.x22, action)
        next_state = vec_moves[0]  # pick first legal action
        return next_state

    def step_p2(self, action):
        vec_moves = get_legal_moves(self.x21, self.x22, self.x11, self.x12, action)
        next_state = vec_moves[0]  # pick first legal action
        return next_state

    def get_score(self):
        return sum(x == RoyalGameOfUr.x_final for x in [self.x11, self.x12])

    def step(self, action=None):
        old_score = self.get_score()

        # Player 1 moves
        do_roll = True
        first_roll = True

        while do_roll:
            if first_roll and action is not None:
                action_p1 = action
            else:
                action_p1 = roll_4_dice()

            first_roll = False

            old_state = [self.x11, self.x12, self.x21, self.x22]

            next_state = self.step_p1(action_p1)
            self.x11, self.x12, self.x21, self.x22 = next_state

            do_roll = (
                (self.x11 in RoyalGameOfUr.vec_x_star and self.x11 != old_state[0]) or
                (self.x12 in RoyalGameOfUr.vec_x_star and self.x12 != old_state[1])
            )

        # reward from Player 1's full turn
        new_score = self.get_score()
        reward = new_score - old_score

        # Check whether Player 1 won before Player 2 moves
        if self.x11 == RoyalGameOfUr.x_final and self.x12 == RoyalGameOfUr.x_final:
            self.game_over = True
            self.counter += 1
            return [self.x11, self.x12, self.x21, self.x22], reward

        # Player 2 moves
        do_roll = True

        while do_roll:
            action_p2 = roll_4_dice()

            old_state = [self.x11, self.x12, self.x21, self.x22]

            next_state = self.step_p2(action_p2)
            self.x21, self.x22, self.x11, self.x12 = next_state

            do_roll = (
                (self.x21 in RoyalGameOfUr.vec_x_star and self.x21 != old_state[2]) or
                (self.x22 in RoyalGameOfUr.vec_x_star and self.x22 != old_state[3])
            )

        # Check whether Player 2 won
        self.game_over = (
            (self.x11 == RoyalGameOfUr.x_final and self.x12 == RoyalGameOfUr.x_final) or
            (self.x21 == RoyalGameOfUr.x_final and self.x22 == RoyalGameOfUr.x_final)
        )

        self.counter += 1

        return [self.x11, self.x12, self.x21, self.x22], reward

    def render(self, player1_loc, player2_loc):
        """
        The input for this render functions are lists or sets player1_loc and player2_loc.
        These should contain the location of the pieces for the respective player.
        The location should be denoted by the number of the square on the path of that player.
        More precisely, we can number the squares for both players on their path from 1 to 14, with 1 the first square on the
        path, and 14 the last square on the path, which is the final one before leaving the board.
        If player1_loc then equals [2, 4] for example, then he has two pieces in play. The first is on square 2, i.e., the
        second square on their path, while the second is on square 4, i.e., the fourth square on their path.
        """

        p1_score = np.sum(np.array(player1_loc) == RoyalGameOfUr.x_final)
        p2_score = np.sum(np.array(player2_loc) == RoyalGameOfUr.x_final)
        print(f"\n[render] round {self.counter}. P1 score {p1_score}, P2 score {p2_score}")

        # clean input
        player1_loc = [i for i in player1_loc if 1 <= i <= 14]
        player2_loc = [i for i in player2_loc if 1 <= i <= 14]

        # safe zone
        for i in range(4):  # 0...3
            loc_curr = i + 1
            base_str = f"{loc_curr:02d}{"*" if loc_curr in RoyalGameOfUr.vec_x_star else ":"}"
            if loc_curr in player1_loc and loc_curr in player2_loc:
                print(f"{base_str} P1 P2")
            elif loc_curr in player1_loc:
                print(f"{base_str} P1")
            elif loc_curr in player2_loc:
                print(f"{base_str}    P2")
            else:
                print(f"{base_str}")

        # shared/battle zone
        for i in range(4, 12):  # 4...11
            loc_curr = i+1
            base_str = f"{loc_curr:02d}{"*" if loc_curr in RoyalGameOfUr.vec_x_star else ":"}"
            if loc_curr in player1_loc and loc_curr in player2_loc:
                raise Exception("there cannot be 2 pieces here")
            elif loc_curr in player1_loc:
                print(f"{base_str} P1")
            elif loc_curr in player2_loc:
                print(f"{base_str} P2")
            else:
                print(f"{base_str}")

        # safe zone
        for i in range(12, 14):  # 12...13
            loc_curr = i+1
            base_str = f"{loc_curr:02d}{"*" if loc_curr in RoyalGameOfUr.vec_x_star else ":"}"
            if loc_curr in player1_loc and loc_curr in player2_loc:
                print(f"{base_str} P1 P2")
            elif loc_curr in player1_loc:
                print(f"{base_str} P1")
            elif loc_curr in player2_loc:
                print(f"{base_str} P2")
            else:
                print(f"{base_str}")

        return


def sarsa(env, num_episodes=10000, alpha=0.1, epsilon=0.1, n=10, lamb=0.5):
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


if __name__ == "__main__":
    # Play the royal game of ur
    N = 2
    ur = RoyalGameOfUr(N)

    # ur.render([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [1, 4, 8]) # this must throw error. 2 pieces in battle zone.
    # ur.render([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [1, 2, 3, 4, 13, 14])  # full board
    # ur.render([1, 2, 3, 4, 13, 14], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # full board

    while not ur.game_over:
        next_state, _ = ur.step()
        ur.render([ur.x11, ur.x12], [ur.x21, ur.x22])

        # if ur.counter == 3:
        #     exit(0)

    #     # time.sleep(0.5)

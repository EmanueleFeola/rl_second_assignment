from utils import append_move, roll_4_dice, get_legal_moves
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RoyalGameOfUr(gym.Env):
    x_final = 15
    vec_x_star = [4, 8, 14]

    def __init__(self, n_piece):
        self.n_piece = n_piece  # Number of pieces per player
        self.state_space = spaces.Discrete(self.n_piece)
        self.action_space = spaces.Discrete(4)

        # define the states of the pieces s=(x11, ..., x22, dice_sum)
        self.x11 = None  # piece 1 of player 1
        self.x12 = None  # piece 2 of player 1
        self.x21 = None  # piece 1 of player 2
        self.x22 = None  # piece 2 of player 2
        self.dice_sum = None

        self.counter = None
        self.game_over = None

        self.reset()

    def reset(self):
        # reset all pieces to zero, i.e., not on the board.
        self.x11 = 0
        self.x12 = 0
        self.x21 = 0
        self.x22 = 0
        self.game_over = False
        self.counter = 0

        return [self.x11, self.x12, self.x21, self.x22]

    def step_p1(self, action):
        vec_moves = get_legal_moves(self.x11, self.x12, self.x21, self.x22, action, RoyalGameOfUr.x_final)
        next_state = vec_moves[0]  # pick first legal action
        return next_state

    def step_p2(self, action):
        vec_moves = get_legal_moves(self.x21, self.x22, self.x11, self.x12, action, RoyalGameOfUr.x_final)
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
            return [self.x11, self.x12, self.x21, self.x22], reward, self.game_over, {}

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

        return [self.x11, self.x12, self.x21, self.x22], reward, self.game_over, {}

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

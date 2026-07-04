from utils import roll_4_dice
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RoyalGameOfUr(gym.Env):
    vec_x_star = [4, 8, 14]
    vec_x_shared = [5, 6, 7, 8, 9, 10, 11, 12]
    vec_x_shared_safe = [8]
    x_final = 15

    def __init__(self, n_piece, flag_verbose=False):
        # self.n_piece = n_piece  # assume 2 pieces only
        self.observation_space = spaces.MultiDiscrete([
            16,  # x11: 0...15
            16,  # x12: 0...15
            16,  # x21: 0...15
            16,  # x22: 0...15
            5    # dice_sum: 0...4
        ])
        self.action_space = spaces.Discrete(3)  # move piece 1, move piece 2, no legal move (pass)

        # define the states of the pieces s=(x11, ..., x22, dice_sum)
        self.x11 = None  # piece 1 of player 1
        self.x12 = None  # piece 2 of player 1
        self.x21 = None  # piece 1 of player 2
        self.x22 = None  # piece 2 of player 2
        self.dice_sum = None

        self.move_count = None
        self.game_over = None
        self.flag_verbose = flag_verbose

        self.reset()

    def get_state(self):
        return (self.x11, self.x12, self.x21, self.x22, self.dice_sum)

    def get_score(self):
        _sum = 0
        if self.x11 == RoyalGameOfUr.x_final:
            _sum = _sum + 1
        if self.x12 == RoyalGameOfUr.x_final:
            _sum = _sum + 1
        return _sum

    def reset(self):
        # state definition. reset all pieces to zero, i.e., not on the board, and toss the dice for the first turn.
        self.x11 = 0
        self.x12 = 0
        self.x21 = 0
        self.x22 = 0
        self.dice_sum = roll_4_dice()  # dice roll for first turn

        self.game_over = False
        self.move_count = 0

        return self.get_state()

    def get_valid_actions(self, flag_p1_turn):
        """
        Given current state (x11, x12, x21, x22) and dice_value,
        return all legal actions for the active player.

        State convention:
            x11, x12 = active player's pieces
            x21, x22 = opponent's pieces

        Position convention:
            0  = start
            15 = final / completed
            1..4   = active player's private entry path
            5..12  = shared battle path
            13..14 = active player's private exit path

        Action convention:
            0 = move active player's first piece
            1 = move active player's second piece
            2 = pass / no legal move

        Rules:
            - Dice value 0 means no movement.
            - Finished pieces cannot move.
            - A piece cannot move beyond x_final.
            - Scoring requires exact dice value.
            - A piece cannot land on the active player's other piece, except at final.
            - Captures happen only on shared squares 5..12.
            - Square 8 is protected, so landing on an opponent there is illegal.
            - Opponent pieces outside the shared path do not block the active player,
            because those are private paths.
        """

        MOVE_PIECE_1 = 0
        MOVE_PIECE_2 = 1
        PASS = 2

        vec_valid_action = []
        vec_x_p1 = [self.x11, self.x12]
        vec_x_p2 = [self.x21, self.x22]

        if not flag_p1_turn:
            vec_x_p1 = [self.x21, self.x22]
            vec_x_p2 = [self.x11, self.x12]

        if self.dice_sum == 0:
            return [PASS]

        for piece_idx in range(2):
            x_curr = vec_x_p1[piece_idx]
            x_other = vec_x_p1[1 - piece_idx]

            # Finished pieces cannot move.
            if x_curr == self.x_final:
                continue

            x_next = x_curr + self.dice_sum

            # Cannot move beyond final.
            if x_next > self.x_final:
                continue

            # Cannot land on own other piece, except at final.
            if x_next != self.x_final and x_next == x_other:
                continue

            # Reaching final is legal if exact.
            if x_next == self.x_final:
                vec_valid_action.append(piece_idx)
                continue

            # Opponent collision matters only on shared squares.
            if x_next in self.vec_x_shared and x_next in vec_x_p2:

                # Cannot capture or land on protected shared square.
                if x_next in self.vec_x_shared_safe:
                    continue

                # Capture on shared non-protected square is legal.
                vec_valid_action.append(piece_idx)
                continue

            # If x_next is outside shared path, opponent pieces do not matter.
            # If x_next is shared but empty, it is also legal.
            vec_valid_action.append(piece_idx)

        if len(vec_valid_action) == 0:
            return [PASS]

        return vec_valid_action

    def apply_action(self, action, flag_p1_turn):
        PASS = 2

        if action == PASS:
            return False

        if flag_p1_turn:
            active = [self.x11, self.x12]
            opponent = [self.x21, self.x22]
        else:
            active = [self.x21, self.x22]
            opponent = [self.x11, self.x12]

        x_old = active[action]
        x_new = x_old + self.dice_sum
        active[action] = x_new

        # Capture only with the moved piece, only in shared non-protected squares.
        if x_new in self.vec_x_shared and x_new not in self.vec_x_shared_safe:
            if opponent[0] == x_new:
                opponent[0] = 0
            elif opponent[1] == x_new:
                opponent[1] = 0

        if flag_p1_turn:
            self.x11, self.x12 = active
            self.x21, self.x22 = opponent
        else:
            self.x21, self.x22 = active
            self.x11, self.x12 = opponent

        return x_new in self.vec_x_star

    def _step_end(self, reward):
        self.dice_sum = roll_4_dice()
        return self.get_state(), reward, self.game_over, {}

    def step(self, action):
        old_score = self.get_score()
        # old_state = [self.x11, self.x12, self.x21, self.x22]
        self.move_count += 1

        ##################
        # P1 moves with action chosen by agent
        if self.flag_verbose:
            print(f"[step] P1 moves {action} by {self.dice_sum} steps")
        landed_on_star = self.apply_action(action, flag_p1_turn=True)

        # reward from action
        new_score = self.get_score()
        reward = new_score - old_score

        # check game over
        if self.x11 == RoyalGameOfUr.x_final and self.x12 == RoyalGameOfUr.x_final:
            self.game_over = True
            return self.get_state(), reward, self.game_over, {}

        # check double action. if yes, return to agent, i.e., skip p2 move.
        if landed_on_star:
            return self._step_end(reward)

        ##################
        # P2 moves with random action
        flag_p2_turn = True

        while (flag_p2_turn):
            # old_state = [self.x11, self.x12, self.x21, self.x22]
            self.dice_sum = roll_4_dice()
            vec_legal_action = self.get_valid_actions(flag_p1_turn=False)

            # choose action in trivial manner
            action_p2 = np.random.choice(vec_legal_action)
            landed_on_star_p2 = self.apply_action(action_p2, flag_p1_turn=False)
            if self.flag_verbose:
                print(f"[step] P2 moves {action_p2} by {self.dice_sum} steps")

            # check double action for p2
            if not landed_on_star_p2:
                flag_p2_turn = False

        # check game over
        if self.x21 == RoyalGameOfUr.x_final and self.x22 == RoyalGameOfUr.x_final:
            self.game_over = True
            return self._step_end(reward)

        return self._step_end(reward)

    def render(self, player1_loc, player2_loc):
        p1_score = np.sum(np.array(player1_loc) == RoyalGameOfUr.x_final)
        p2_score = np.sum(np.array(player2_loc) == RoyalGameOfUr.x_final)
        print(f"\n[render] round {self.move_count}. P1 score {p1_score}, P2 score {p2_score}")

        # clean input
        player1_loc = [i for i in player1_loc if 1 <= i <= 14]
        player2_loc = [i for i in player2_loc if 1 <= i <= 14]

        # safe zone
        for i in range(4):  # 0...3
            loc_curr = i + 1
            base_str = f"{loc_curr:02d}{'*' if loc_curr in RoyalGameOfUr.vec_x_star else ':'}"
            if loc_curr in player1_loc and loc_curr in player2_loc:
                print(f"{base_str} P1 P2")
            elif loc_curr in player1_loc:
                print(f"{base_str} P1")
            elif loc_curr in player2_loc:
                print(f"{base_str}    P2")
            else:
                print(f"{base_str}")

        # shared/battle zone
        print("---battle zone---")
        for i in range(4, 12):  # 4...11
            loc_curr = i+1
            base_str = f"{loc_curr:02d}{'*' if loc_curr in RoyalGameOfUr.vec_x_star else ':'}"
            if loc_curr in player1_loc and loc_curr in player2_loc:
                raise Exception("there cannot be 2 pieces here")
            elif loc_curr in player1_loc:
                print(f"{base_str} P1")
            elif loc_curr in player2_loc:
                print(f"{base_str} P2")
            else:
                print(f"{base_str}")
        print("-----------")

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

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


def get_valid_actions(x11, x12, x21, x22, dice_value, x_final):
    """
    Given current state (x11, x12, x21, x22) and dice_value,
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
    # print(f"[get_valid_actions] dice_value {dice_value}")

    vec_moves = []

    # x_final = RoyalGameOfUr.x_final
    shared_squares = range(5, 13)   # 5, ..., 12
    protected_shared_squares = [8]  # shared rosette/star square

    # If dice_value is 0, no piece moves.
    if dice_value == 0:
        return [[x11, x12, x21, x22]]

    vec_p1 = [x11, x12]

    for i in range(2):
        x_curr = vec_p1[i]
        x_other = vec_p1[1 - i]

        # Finished pieces cannot move.
        if x_curr == x_final:
            continue

        x_next = x_curr + dice_value

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

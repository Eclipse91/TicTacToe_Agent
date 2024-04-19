import logging
import numpy as np


# Rewards for each strike
HIT_REWARD = 1
REPEAT_STRIKE_REWARD = -1
# Reward for finishing the game within MAX_STEPS_PER_EPISODE
FINISHED_GAME_REWARD = 10
FINISHED_GAME_REWARD_DRAW = 5
# Reward for not finishing the game within MAX_STEPS_PER_EPISODE
UNFINISHED_GAME_REWARD = -10


class TicTacToeEnvironment:
    def __init__(self):
        self.state = np.zeros((3, 3))  # Initialize the board
        self.current_player = 1  # Player 1 starts
        self.moves = 0

    def _reset(self):
        self.state = np.zeros((3, 3))  # Reset the board
        self.current_player = 1  # Player 1 starts
        self.moves = 0
        return self.state

    def _step(self, action):
        row, col = action // 3, action % 3
        if self.state[row, col] == 0:  # Check if the position is empty
            self.state[row, col] = self.current_player  # Place the mark
            logging.info(
              f'\nPlayer {self.current_player}\tAction: {action}\n{self.state}'
            )
            winner = self.check_winner()

            if winner is not None:
                logging.info(f'WON {self.current_player}\n{self.state}')
                reward = FINISHED_GAME_REWARD #if winner == 1 else -1
                return self.state, reward, True  # Game over
            elif np.all(self.state != 0):
                logging.info(f'DRAW:\n{self.state}')
                return self.state, FINISHED_GAME_REWARD_DRAW, True  # Draw
            else:
                # Switch players
                self.current_player = 2 if self.current_player == 1 else 1 
                self.moves += 1
                if self.moves >= 9:
                    return self.state, -10, True
                return self.state, HIT_REWARD, False  # Continue game
        else:
            self.moves += 1
            if self.moves >= 9:
                return self.state, UNFINISHED_GAME_REWARD, True
            return self.state, REPEAT_STRIKE_REWARD, False  # Invalid move

    def check_winner(self):
        for player in [1, 2]:
            if np.any(np.all(self.state == player, axis=0)) \
                or np.any(np.all(self.state == player, axis=1)):
                return player
            if np.all(np.diag(self.state) == player) \
                or np.all(np.diag(np.fliplr(self.state)) == player):
                return player
        return None
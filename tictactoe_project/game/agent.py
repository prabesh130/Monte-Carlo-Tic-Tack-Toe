import os
import pickle
import random
import numpy as np
from collections import defaultdict


class TicTacToe2D:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return tuple(self.board.flatten())

    def available_actions(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def check_winner(self):
        board = self.board

        # Check rows
        for row in board:
            s = int(row.sum())
            if abs(s) == 3:
                return 1 if s == 3 else -1

        # Check columns
        for col in range(3):
            s = int(board[:, col].sum())
            if abs(s) == 3:
                return 1 if s == 3 else -1

        # Check diagonals
        diag1 = int(board.trace())
        if abs(diag1) == 3:
            return 1 if diag1 == 3 else -1

        diag2 = int(board[0, 2] + board[1, 1] + board[2, 0])
        if abs(diag2) == 3:
            return 1 if diag2 == 3 else -1

        # Draw
        if not (board == 0).any():
            return 0

        return None

    def step(self, action):
        r, c = action
        self.board[r, c] = self.current_player
        winner = self.check_winner()
        done = winner is not None

        if done:
            if winner == 1:
                reward = 1
            elif winner == -1:
                reward = -1
            else:
                reward = 0
        else:
            reward = 0

        # Switch player
        self.current_player *= -1
        return tuple(self.board.flatten()), reward, done

    # ---------- Static helper methods ----------

    @staticmethod
    def epsilon_greedy_action(Q, state, avail_actions, epsilon):
        if random.random() < epsilon:
            return random.choice(avail_actions)
        qvals = Q[state]
        best_val = max(qvals[a] for a in avail_actions)
        best_actions = [a for a in avail_actions if qvals[a] == best_val]
        return random.choice(best_actions)

    @staticmethod
    def can_win_next_move(board, player):
        for r in range(3):
            for c in range(3):
                if board[r, c] == 0:
                    temp = board.copy()
                    temp[r, c] = player
                    env_temp = TicTacToe2D()
                    env_temp.board = temp
                    if env_temp.check_winner() == player:
                        return (r, c)
        return None

    @staticmethod
    def get_immediate_reward(old_board, action, new_board, player):
        r, c = action
        reward = 0.0

        # Reward for winning move
        env_temp = TicTacToe2D()
        env_temp.board = new_board.copy()
        winner = env_temp.check_winner()
        if winner == player:
            return 10.0

        # Reward for blocking
        opponent = -player
        if TicTacToe2D.can_win_next_move(old_board, opponent):
            if TicTacToe2D.can_win_next_move(new_board, opponent):
                reward -= 2.0  # failed to block
            else:
                reward += 10.0  # successfully blocked

        # Center & corners are preferred
        if (r, c) == (1, 1):
            reward += 0.1
        elif (r, c) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            reward += 0.05

        return reward

    @staticmethod
    def semi_skilled_opponent(board):
        agent = 1
        block_move = TicTacToe2D.can_win_next_move(board, agent)
        if block_move:
            return block_move
        moves = [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]
        return random.choice(moves)

    @staticmethod
    def generate_episode(env, Q, epsilon):
        env.reset()
        episode = []
        state = tuple(env.board.flatten())
        done = False

        while not done:
            avail = env.available_actions()
            if env.current_player == 1:
                # Agent's turn
                action = TicTacToe2D.epsilon_greedy_action(Q, state, avail, epsilon)
                old_board = env.board.copy()
                next_state, reward, done = env.step(action)
                imm_r = TicTacToe2D.get_immediate_reward(old_board, action, env.board, 1)
                episode.append((tuple(old_board.flatten()), action, 1, imm_r))
                state = next_state
            else:
                # Opponent turn (semi-skilled 70%, random 30%)
                if random.random() < 0.7:
                    action = TicTacToe2D.semi_skilled_opponent(env.board)
                else:
                    action = random.choice(avail)

                old_board = env.board.copy()
                next_state, _, done = env.step(action)
                episode.append((tuple(old_board.flatten()), action, -1, 0.0))
                state = next_state

        final = env.check_winner()
        final_reward = 10.0 if final == 1 else -10.0 if final == -1 else 0.0

        # Add final reward to agent’s moves
        enhanced = []
        for s, a, player, imm_r in episode:
            total_r = imm_r + final_reward if player == 1 else 0.0
            enhanced.append((s, a, player, total_r))

        return enhanced, final


# ---------- Trainer wrapper ----------

class Trainer:
    def __init__(self, model_path="model.pkl"):
        self.env = TicTacToe2D()
        self.Q = defaultdict(lambda: defaultdict(float))
        self.model_path = model_path
        if os.path.exists(model_path):
            self.load_model()

    def train(self, episodes=10000, epsilon=0.1, alpha=0.1):
        for _ in range(episodes):
            episode, _ = TicTacToe2D.generate_episode(self.env, self.Q, epsilon)
            for state, action, player, reward in episode:
                if player == 1:  # update agent moves
                    old_val = self.Q[state].get(action, 0.0)
                    new_val = old_val + alpha * (reward - old_val)
                    self.Q[state][action] = new_val
        self.save_model()
    def train_one_episode(self,epsilon,alpha):
        episode,final_winner=TicTacToe2D.generate_episode(self.env,self.Q,epsilon)
        episode_data=[]
        for state,action,player,reward in episode:
            if player==1: 
                old_val=self.Q[state].get(action,0.0)
                new_val=old_val+alpha*(reward-old_val)
                self.Q[state][action]=new_val
            board_state = np.array(state).reshape(3, 3)
            episode_data.append((board_state, action, player, reward))
        return episode_data,final_winner

    def act(self, board, epsilon=0.0):
        """Choose best move for given board state."""
        state = tuple(board.flatten())
        avail = [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]
        if not avail:
            return None
        return TicTacToe2D.epsilon_greedy_action(self.Q, state, avail, epsilon)

    def save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load_model(self):
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
            self.Q = defaultdict(lambda: defaultdict(float), data)

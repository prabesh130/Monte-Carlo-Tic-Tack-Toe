import tkinter as tk
from tkinter import ttk
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

# ---------------- Tic-Tac-Toe Environment ----------------
class TicTacToe2D:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return tuple(self.board.flatten())

    def available_actions(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def step(self, action):
        r, c = action
        self.board[r, c] = self.current_player
        winner = self.check_winner()
        done = winner is not None or not (self.board == 0).any()
        if done:
            if winner == 1: reward = 1
            elif winner == -1: reward = -1
            else: reward = 0
        else:
            reward = 0
        self.current_player *= -1
        return tuple(self.board.flatten()), reward, done

    def check_winner(self):
        b = self.board
        # Check rows
        for row in b:
            s = int(row.sum())
            if abs(s) == 3:
                return 1 if s == 3 else -1

        # Check columns
        for col in range(3):
            column = b[:, col]
            s = int(column.sum())
            if abs(s) == 3:
                return 1 if s == 3 else -1

        # Check diagonals
        diag1 = b.trace()
        if abs(int(diag1)) == 3:
            return 1 if int(diag1) == 3 else -1

        diag2 = int(b[0,2] + b[1,1] + b[2,0])
        if abs(diag2) == 3:
            return 1 if diag2 == 3 else -1

        # Draw
        if not (b == 0).any():
            return 0

        return None

# ---------------- Monte Carlo Helpers ----------------
def epsilon_greedy_action(Q, state, avail_actions, epsilon):
    normaized_state=normailze_state(state,env.current_player)
    if random.random() < epsilon:
        return random.choice(avail_actions)
    qvals = Q[state]
    best_val = max(qvals[a] for a in avail_actions)
    best_actions = [a for a in avail_actions if qvals[a] == best_val]
    return random.choice(best_actions)

def can_win_next_move(board, player):
    """Check if player can win on next move, return winning action or None.
       Does NOT modify the input board."""
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
def normailze_state(state,player):
    if player==-1:
        return tuple(-x for x in state)
    return state
def get_immediate_reward(old_board, action, new_board, player):
    """Give immediate rewards for good/bad moves."""
    r, c = action
    reward = 0.0

    # Reward for winning move
    env_temp = TicTacToe2D()
    env_temp.board = new_board.copy()
    winner = env_temp.check_winner()
    if winner == player:
        return 10.0

    # Reward for blocking opponent's win
    opponent = -player
    if can_win_next_move(old_board, opponent):
        if can_win_next_move(new_board, opponent):
            reward -= 2.0  # didn't block
        else:
            reward += 3.0  # blocked

    # Small preference for center / corners
    if (r, c) == (1, 1):
        reward += 0.1
    elif (r, c) in [(0,0), (0,2), (2,0), (2,2)]:
        reward += 0.05

    return reward

def generate_episode(env, Q, epsilon):
    """Generate a full episode with both players' moves."""
    env.reset()
    episode = []
    state = tuple(env.board.flatten())
    done = False

    while not done:
        avail = env.available_actions()
        if env.current_player == 1:
            # Agent's turn
            action = epsilon_greedy_action(Q, state, avail, epsilon)
            old_board = env.board.copy()
            next_state, reward, done = env.step(action)
            immediate_reward = get_immediate_reward(old_board, action, env.board, 1)
            episode.append((tuple(old_board.flatten()), action, 1, immediate_reward))
            state = next_state
        else:
            # Opponent's turn (random)
            action = random.choice(avail)
            old_board = env.board.copy()
            next_state, reward, done = env.step(action)
            episode.append((tuple(old_board.flatten()), action, -1, 0.0))
            state = next_state

    final = env.check_winner()
    final_reward = 10.0 if final == 1 else -10.0 if final == -1 else 0.0

    # Apply final reward only to agent moves
    enhanced_episode = []
    for s, a, player, imm_r in episode:
        total_reward = imm_r + final_reward if player == 1 else 0.0
        enhanced_episode.append((s, a, player, total_reward))

    return enhanced_episode, final

# ---------------- Training UI ----------------
class TrainingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe Training Visualization")
        self.env = TicTacToe2D()
        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns_sum = defaultdict(lambda: defaultdict(float))
        self.returns_count = defaultdict(lambda: defaultdict(int))

        # Initialize counters and parameters FIRST
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.episode = 0
        self.num_episodes = 20000
        self.epsilon = 0.4
        self.epsilon_decay = 0.995

        # Play mode variables
        self.play_mode = False
        self.human_turn = True
        self.play_env = TicTacToe2D()

        # Create UI elements AFTER initializing variables
        self.create_board()
        self.create_stats_display()

        # Control buttons frame
        button_frame = tk.Frame(root)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)

        self.start_button = tk.Button(button_frame, text="Start Training", command=self.train_step)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.play_button = tk.Button(button_frame, text="Play vs Agent", command=self.start_play_mode, state="disabled")
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(button_frame, text="Reset Training", command=self.reset_training)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        self.save_button=tk.Button(button_frame,text="Save Model",command=self.save_q_table)
        self.save_button.pack(side=tk.LEFT,padx=5)
        self.load_button=tk.Button(button_frame,text="load Model",command=self.load_q_table)
    def save_q_table(self):
        file=filedialog.asksaveasfilename(defaultextension=".pkl",filetypes=[("Pickle Files","*.pkl")])
        if file:
            with open(file, "wb") as f:
                pickle.dump(self.Q,f)
            messagebox.showinfo("saved",f"q-table saved to {file}")
    def load_q_table(self):
        file=filedialog.askopenfilename(filetypes=[("Pickle Files","*.pkl")])
        if file:
            with open(file,"rb") as f:
                self.Q=pickle.load(f)
            messagebox.showinfo("Loaded",f"Q-tbale loaded from {file}")
            self.play_button.config(state="normal")
            self.info_label.config(text="Loaded pre_trained Q-table. Ready to play!")        
            
    def create_board(self):
        self.buttons = {}
        for r in range(3):
            for c in range(3):
                btn = tk.Button(self.root, text=" ", font=("Arial", 24), width=4, height=2,
                               borderwidth=1, relief="solid", command=lambda r=r, c=c: self.human_move(r, c))
                btn.grid(row=r, column=c)
                self.buttons[(r, c)] = btn

    def create_stats_display(self):
        stats_frame = tk.Frame(self.root)
        stats_frame.grid(row=3, column=0, columnspan=3, pady=10)

        self.episode_label = tk.Label(stats_frame, text="Episode: 0", font=("Arial", 12))
        self.episode_label.pack()

        self.wins_label = tk.Label(stats_frame, text="Wins: 0", font=("Arial", 12), fg="green")
        self.wins_label.pack()

        self.losses_label = tk.Label(stats_frame, text="Losses: 0", font=("Arial", 12), fg="red")
        self.losses_label.pack()

        self.draws_label = tk.Label(stats_frame, text="Draws: 0", font=("Arial", 12), fg="blue")
        self.draws_label.pack()

        self.winrate_label = tk.Label(stats_frame, text="Win Rate: 0.0%", font=("Arial", 12), fg="purple")
        self.winrate_label.pack()

        self.epsilon_label = tk.Label(stats_frame, text=f"Epsilon: {self.epsilon:.2f}", font=("Arial", 12), fg="orange")
        self.epsilon_label.pack()

        self.info_label = tk.Label(stats_frame, text="Ready to start training...", font=("Arial", 11))
        self.info_label.pack()

    def update_board(self, use_play_env=False):
        syms = {1: "X", -1: "O", 0: " "}
        board = self.play_env.board if (self.play_mode and use_play_env) else (self.play_env.board if self.play_mode else self.env.board)
        for r in range(3):
            for c in range(3):
                symbol = syms[board[r, c]]
                self.buttons[(r, c)].config(text=symbol)
                
                if self.play_mode:
                    if board[r, c] == 0 and self.human_turn:
                        self.buttons[(r, c)].config(state="normal", bg="lightblue")
                    else:
                        self.buttons[(r, c)].config(state="disabled", bg="SystemButtonFace")
                else:
                    self.buttons[(r, c)].config(state="disabled", bg="SystemButtonFace")

    def update_stats_display(self):
        self.episode_label.config(text=f"Episode: {self.episode}")
        self.wins_label.config(text=f"Wins: {self.wins}")
        self.losses_label.config(text=f"Losses: {self.losses}")
        self.draws_label.config(text=f"Draws: {self.draws}")
        self.epsilon_label.config(text=f"Epsilon: {self.epsilon:.3f}")

        total_games = self.wins + self.losses + self.draws
        if total_games > 0:
            win_rate = (self.wins / total_games) * 100
            self.winrate_label.config(text=f"Win Rate: {win_rate:.1f}%")
        else:
            self.winrate_label.config(text="Win Rate: 0.0%")
    def use_pre_trained_data(self):
        self

    def train_step(self):
        if self.episode >= self.num_episodes:
            self.info_label.config(text=f"Training completed! ({self.num_episodes} episodes)")
            self.start_button.config(text="Training Complete", state="disabled")
            self.play_button.config(state="normal")
            return

        self.episode += 1
        self.episode_data, game_result = generate_episode(self.env, self.Q, self.epsilon)

        # Update counters
        if game_result == 1:
            self.wins += 1
        elif game_result == -1:
            self.losses += 1
        else:
            self.draws += 1

        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

        # Update display
        self.update_stats_display()
        self.info_label.config(text=f"Training... Episode {self.episode}")

        # Animate episode
        self.current_move_index = 0
        self.env.reset()
        self.play_episode_step()

    def play_episode_step(self):
        if self.current_move_index >= len(self.episode_data):
            # Update Q-values for agent moves only
            visited = set()
            for s, a, player, r in self.episode_data:
                if player == 1:  # Only update for agent
                    if (s, a) not in visited:
                        visited.add((s, a))
                        self.returns_sum[s][a] += r
                        self.returns_count[s][a] += 1
                        self.Q[s][a] = self.returns_sum[s][a] / self.returns_count[s][a]

            final_result = self.env.check_winner()
            result_text = "X WINS!" if final_result == 1 else "O WINS!" if final_result == -1 else "DRAW!"
            self.info_label.config(text=f"Game Over: {result_text}")

            # Next episode after brief delay
            self.root.after(50, self.train_step)
            return

        # Apply next move
        state, action, player, reward = self.episode_data[self.current_move_index]
        self.env.board = np.array(state).reshape((3, 3)).copy()
        r, c = action
        self.env.board[r, c] = player

        self.update_board()
        self.current_move_index += 1

        # Show next move after delay
        self.root.after(3, self.play_episode_step)

    def start_play_mode(self):
        """Start interactive play mode against the trained agent"""
        self.play_mode = True
        self.play_env.reset()
        self.human_turn = True
        self.play_button.config(state="disabled")
        self.start_button.config(state="disabled")
        self.info_label.config(text="Your turn! Click a square to play. You are X.")
        self.update_board(use_play_env=True)

        self.exit_play_button = tk.Button(self.root, text="Exit Play Mode", command=self.exit_play_mode)
        self.exit_play_button.grid(row=6, column=0, columnspan=3, pady=5)

    def human_move(self, r, c):
        """Handle human player move"""
        if not self.play_mode or not self.human_turn:
            return

        if self.play_env.board[r, c] != 0:
            return

        # Make human move (X = 1)
        self.play_env.board[r, c] = 1
        self.update_board(use_play_env=True)

        # Check if game is over
        winner = self.play_env.check_winner()
        if winner is not None or not (self.play_env.board == 0).any():
            self.end_play_game(winner)
            return

        # Agent turn
        self.human_turn = False
        self.info_label.config(text="Agent is thinking...")
        
        # Agent uses flipped perspective (sees itself as +1)
        agent_board = -self.play_env.board
        agent_state = tuple(agent_board.flatten())
        avail_actions = [(rr, cc) for rr in range(3) for cc in range(3) if agent_board[rr, cc] == 0]
        
        if not avail_actions:
            self.end_play_game(self.play_env.check_winner())
            return

        # Greedy agent action (epsilon=0)
        action = epsilon_greedy_action(self.Q, agent_state, avail_actions, 0.0)
        ar, ac = action
        self.play_env.board[ar, ac] = -1
        self.update_board(use_play_env=True)

        # Check for end
        winner = self.play_env.check_winner()
        if winner is not None or not (self.play_env.board == 0).any():
            self.end_play_game(winner)
            return

        # Back to human
        self.human_turn = True
        self.info_label.config(text="Your turn! You are X.")
        self.update_board(use_play_env=True)

    def end_play_game(self, winner):
        """End the current play game and show result"""
        if winner == 1:
            result_text = "You won!"
        elif winner == -1:
            result_text = "Agent won!"
        else:
            result_text = "It's a draw!"

        self.info_label.config(text=f"Game Over: {result_text}")

        for btn in self.buttons.values():
            btn.config(state="disabled")

        self.new_game_button = tk.Button(self.root, text="New Game", command=self.new_play_game)
        self.new_game_button.grid(row=7, column=0, columnspan=3, pady=5)

    def new_play_game(self):
        """Start a new game in play mode"""
        self.play_env.reset()
        self.human_turn = True
        self.info_label.config(text="New game! Your turn. You are X.")
        self.update_board(use_play_env=True)

        if hasattr(self, 'new_game_button'):
            self.new_game_button.destroy()

    def exit_play_mode(self):
        """Exit play mode and return to training interface"""
        self.play_mode = False
        self.play_button.config(state="normal")
        if self.episode < self.num_episodes:
            self.start_button.config(state="normal")
        self.info_label.config(text="Play mode exited. Ready for more training or play.")

        self.env.reset()
        self.update_board(use_play_env=False)

        if hasattr(self, 'exit_play_button'):
            self.exit_play_button.destroy()
        if hasattr(self, 'new_game_button'):
            self.new_game_button.destroy()

    def reset_training(self):
        """Reset all training progress"""
        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns_sum = defaultdict(lambda: defaultdict(float))
        self.returns_count = defaultdict(lambda: defaultdict(int))
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.episode = 0
        self.epsilon = 0.3

        self.update_stats_display()
        self.start_button.config(text="Start Training", state="normal")
        self.play_button.config(state="disabled")
        self.info_label.config(text="Training reset. Ready to start fresh!")

        self.env.reset()
        self.update_board(use_play_env=False)

# ---------------- Run UI ----------------

if __name__ == "__main__":
    root = tk.Tk()
    ui = TrainingUI(root)
    root.mainloop()
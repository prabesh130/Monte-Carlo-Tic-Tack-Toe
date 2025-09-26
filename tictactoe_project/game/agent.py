import pickle , os ,time
import random
import numpy as np
import pandas as pd


class TictacToe2D:
    def __init__(self):
        self.reset()
    def reset(self):
        self.board=np.zeros((3,3),dtype=int)
        self.current_player=1
        return tuple(self.board.flatten())
    def avail_action(self):

        return [(r,c) for r in range(3) for c in range(3) if self.board[r,c]==0]
    def check_winner(self): 
# views.py
from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string
import numpy as np

from .agent import Trainer  # import your trainer

# Game state
board = np.zeros((3, 3), dtype=int)
current_player = 1
trainer = Trainer()   # load existing model if available


def board_view(request):
    return render(request, "game/board.html", {"board": board})


def make_move(request, r, c):
    global current_player, board, trainer

    r, c = int(r), int(c)

    # human move only if empty
    if board[r, c] == 0:
        board[r, c] = current_player
        current_player = 2 if current_player == 1 else 1

        # if it's agent's turn (O), let the model act
        if current_player == 2:
            move = trainer.act(board, epsilon=0.0)
            if move:
                rr, cc = move
                board[rr, cc] = current_player
                current_player = 1

    html = render_to_string("game/board.html", {"board": board}, request=request)
    return HttpResponse(html)


def start_training(request):
    global trainer
    trainer.train(episodes=5000, epsilon=0.1, alpha=0.1)
    return HttpResponse("<p>Training complete! Model saved.</p>")

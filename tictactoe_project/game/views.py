# views.py
from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string
import numpy as np
from collections import defaultdict
from django.http import JsonResponse
import random

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


Q=defaultdict(lambda:defaultdict(float))
wins=loses=draws=episode=0
epsilon=0.3
num_episode=1000
def train_step(request):
    global Q,wins,losses,draws,episode,epsilon
    if episode>=num_episode:
        return JsonResponse({
            "episode":episode,
            "wins":wins,
            "losses":losses,
            "draws":draws,
            "episode":episode,
            "status":f"Training complete after {num_episode} episodes!",
            "training_done":True
        })
    result=random.random().choice([1,-1,0])
    if result==1:win+=1
    elif result==-1: losses+=1
    else: draws+=1

    episode+=1
    epsilon=max(0.01,epsilon*0.995)

    return JsonResponse({
          "episode": episode,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "epsilon": epsilon,
        "status": f"Training... episode {episode}",
        "training_done": False
    })
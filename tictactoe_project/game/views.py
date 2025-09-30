from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string

# 0 = empty, 1 = X, 2 = O
board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

current_player = 1  # start with X


def board_view(request):
    return render(request, "game/board.html", {"board": board})


def make_move(request, r, c):
    global current_player, board

    r, c = int(r), int(c)

    # only allow move if cell is empty
    if board[r][c] == 0:
        board[r][c] = current_player
        # switch player
        current_player = 2 if current_player == 1 else 1

    html = render_to_string("board.html", {"board": board}, request=request)
    return HttpResponse(html)

from django.shortcuts import render
from .agent import TicTacToe2D
import random
env=TicTacToe2D()

def board_view(request):
    board=request.session.get('board',[[''for _ in range(3)] for _ in range(3)])

    context={
        'board':board,
        'range':range(3),
    }
    return render(request,"game/board.html",context)
def make_move(request,row,col):
    if env.board[row][col]==0:
        env.step((row,col))
        if env.current_player==-1 and not env.check_winner():
            moves=env.available_actions()
            action=random.choce(moves)
            env.step(action)
    context={"board":env.board.tolist(),"current_player":env.current_player}
    return render(request,"game/board.html",context)
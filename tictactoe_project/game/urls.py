from django.urls import path 
from . import views
urlpatterns=[
    path("",views.board_view,name="board"),
    path("move/<int:row>/<int:col>/",views.make_move,name="make_move"),
]
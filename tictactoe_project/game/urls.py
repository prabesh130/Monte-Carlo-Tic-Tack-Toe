from django.urls import path 
from . import views

urlpatterns = [
    path("", views.board_view, name="board"),
    path("move/<int:row>/<int:col>/", views.make_move, name="make_move"),
    
    # Fixed URLs to match template calls
    path("train-step/", views.train_step_html, name='train_step_html'),
    path("start-training/", views.start_training, name="start_training"),
    path("animate-step/", views.animate_episode_step, name='animate_episode_step'),
]
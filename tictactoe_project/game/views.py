from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string
import numpy as np
from django.http import HttpResponse
from .agent import Trainer  # Import your trainer

# --- Global State for the Game ---
# Player 1 is 'X' and Player -1 is 'O' to match the agent logic.
board = np.zeros((3, 3), dtype=int)
current_player = 1
trainer = Trainer()  # Load existing model if available

# --- Global State for Training ---
training_state = {
    "wins": 0,
    "losses": 0,
    "draws": 0,
    "episode": 0,
    "num_episodes": 5000,
    "alpha": 0.1,
    "epsilon": 0.5,
    "min_epsilon": 0.01,
    "decay_rate": 0.9995
}


def board_view(request):
    """Renders the main game board."""
    # Note: Using 1/-1 internally, but need to pass 1/2 to the template for X/O display
    # We pass the raw board and assume the template handles the conversion/display.
    return render(request, "game/board.html", {"board": board})


def make_move(request, r, c):
    """Handles human move (X) and agent response (O)."""
    global current_player, board, trainer

    r, c = int(r), int(c)

    # Human move (Player 1 is X) only if cell is empty
    if board[r, c] == 0 and current_player == 1:
        board[r, c] = current_player
        current_player = -1  # Switch to agent (O)

        # Agent's turn (Player -1 is O)
        if current_player == -1:
            # The agent (Trainer) is trained as player 1. To make it play as -1, 
            # we flip the board state to make it look like a player 1 move.
            temp_board = board * -1
            move = trainer.act(temp_board, epsilon=0.0)

            # If a valid move is returned, the agent acts
            if move:
                rr, cc = move
                board[rr, cc] = current_player  # Agent makes move as -1 (O)
                current_player = 1  # Switch back to human (X)

    # HTMX response: render and swap the board element
    html = render_to_string("game/board.html", {"board": board}, request=request)
    return HttpResponse(html)


def start_training(request):
    """
    Initial view called by the 'Start Training' button. 
    It returns the HTML fragment that starts the HTMX polling loop.
    """
    global training_state
    
    # Reset stats on button click
    training_state.update({
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "episode": 0,
        "epsilon": 0.5,
    })

    # This view renders the template containing hx-get/hx-trigger
    return render(request, "game/training_polling.html", {
        "stats": training_state,
        "status": "Training initiated... Polling server.",
    })


def train_step_html(request):
    """
    The view that is repeatedly called by the HTMX polling loop.
    Performs one episode of Q-learning and returns the updated status HTML.
    """
    global trainer, training_state

    # --- 1. Check if training is complete ---
    if training_state["episode"] >= training_state["num_episodes"]:
        trainer.save_model()  # Final save

        # Renders the 'finished' template to stop the polling loop
        response = render(request, "game/training_finished.html", {
            "stats": training_state,
            "status": f"Training complete after {training_state['num_episodes']} episodes! Model saved.",
        })
        # HTMX header to signal completion and allow button re-enabling
        response['HX-Trigger'] = 'trainingFinished' 
        return response

    # --- 2. Perform one training episode ---
    
    # Decay Epsilon
    epsilon = training_state["epsilon"] * training_state["decay_rate"]
    training_state["epsilon"] = max(training_state["min_epsilon"], epsilon)

    try:
        # Assumes trainer.train_one_episode is defined in agent.py
        final_winner = trainer.train_one_episode(
            epsilon=training_state["epsilon"], 
            alpha=training_state["alpha"]
        )
    except AttributeError:
         # If trainer.train_one_episode is missing
         return HttpResponse("<div id='training-container'>ERROR: Trainer class is missing train_one_episode method.</div>")


    # Update Stats
    if final_winner == 1:  # Agent (Player 1) wins
        training_state["wins"] += 1
    elif final_winner == -1:  # Opponent (Player -1) wins
        training_state["losses"] += 1
    else:  # Draw
        training_state["draws"] += 1
        
    training_state["episode"] += 1

    # --- 3. Return the updated HTML fragment (keeps the polling loop alive) ---
    return render(request, "game/training_polling.html", {
        "stats": training_state,
        "status": f"Training... episode {training_state['episode']} of {training_state['num_episodes']}",
    })
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template.loader import render_to_string
import numpy as np
from .agent import Trainer 

# --- Global State for the Game ---
board = np.zeros((3, 3), dtype=int)
current_player = 1
trainer = Trainer() 
current_episode_data = [] # Stores episode moves
current_move_index = 0    # Current index in the episode data
is_animating = False      # Flag to control the animation polling

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
    return render(request, "game/board.html", {"board": board})


def make_move(request, r, c):
    """Handles human move (X) and agent response (O)."""
    global current_player, board, trainer

    r, c = int(r), int(c)

    # Human move (Player 1 is X) only if cell is empty
    if board[r, c] == 0 and current_player == 1:
        board[r, c] = current_player
        current_player = -1 

        # Agent's turn (Player -1 is O)
        if current_player == -1:
            temp_board = board * -1
            move = trainer.act(temp_board, epsilon=0.0)

            if move:
                rr, cc = move
                board[rr, cc] = current_player 
                current_player = 1 

    # We must render only the board content to be swapped into the #board div
    return render(request, "game/board_content.html", {"board": board})


def start_training(request):
    """
    Initial view called by the 'Start Training' button. 
    It returns the HTML fragment that starts the HTMX polling loop.
    """
    global training_state, is_animating, board
    
    # Reset stats and state on button click
    training_state.update({
        "wins": 0, "losses": 0, "draws": 0, "episode": 0, "epsilon": 0.5,
    })
    is_animating = False
    board[:] = 0

    # This view renders the template containing hx-get/hx-trigger
    return render(request, "game/training_polling.html", {
        "stats": training_state,
        "status": "Training initiated... Polling server.",
    })


def train_step_html(request):
    """
    Runs one full training episode (Q-update happens inside trainer) 
    and returns a fragment to either continue polling or start animation.
    """
    global trainer, training_state, current_episode_data, current_move_index, is_animating, board

    try:
        print(f"\n=== TRAIN_STEP_HTML CALLED ===")
        print(f"is_animating: {is_animating}")
        print(f"episode: {training_state['episode']}")
        
        # If we're animating, just return stats without triggering more training
        if is_animating:
            print(f"[TRAIN] Currently animating, skipping training")
            return render(request, "game/stats_display_only.html", {
                "stats": training_state,
                "status": f"Animating Episode {training_state['episode']}...",
            })
        
        # --- 1. Check if training is complete ---
        if training_state["episode"] >= training_state["num_episodes"]:
            print(f"[TRAIN] Training complete!")
            trainer.save_model() 

            response = render(request, "game/training_finished.html", {
                "stats": training_state,
                "status": f"Training complete after {training_state['num_episodes']} episodes! Model saved.",
            })
            response['HX-Trigger'] = 'trainingFinished' 
            return response

        # --- 2. Perform one training episode & Get Data ---
        
        print(f"[TRAIN] Starting episode {training_state['episode'] + 1}")
        
        # Decay Epsilon
        epsilon = training_state["epsilon"] * training_state["decay_rate"]
        training_state["epsilon"] = max(training_state["min_epsilon"], epsilon)
        
        print(f"[TRAIN] Calling trainer.train_one_episode with epsilon={epsilon:.4f}")

        episode_data, final_winner = trainer.train_one_episode(
            epsilon=training_state["epsilon"], 
            alpha=training_state["alpha"]
        )
        
        print(f"[TRAIN] Episode complete: winner={final_winner}, moves={len(episode_data)}")

        # Update Stats
        if final_winner == 1: 
            training_state["wins"] += 1
        elif final_winner == -1: 
            training_state["losses"] += 1
        else: 
            training_state["draws"] += 1
            
        training_state["episode"] += 1

        # --- 3. Start Animation (always animate for debugging) ---
        current_episode_data = episode_data
        current_move_index = 0
        is_animating = True
        board[:] = 0  # Clear the board before animation starts
        
        print(f"[TRAIN] Episode {training_state['episode']} complete, starting animation with {len(episode_data)} moves")
        print(f"[TRAIN] Returning animation_trigger.html template")
        
        # Return HTML with embedded board that will start polling
        return render(request, "game/animation_trigger.html", {
            "stats": training_state,
            "status": f"Episode {training_state['episode']} finished. Starting animation...",
        })
        
    except Exception as e:
        print(f"\n!!! EXCEPTION IN train_step_html !!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return HttpResponse(f"<div id='training-container' style='color: red; padding: 20px;'>ERROR: {str(e)}<br><pre>{traceback.format_exc()}</pre></div>")


def animate_episode_step(request):
    """
    Called repeatedly by HTMX to display the next move in the episode.
    Returns both board and stats updates.
    """
    global current_episode_data, current_move_index, is_animating, training_state, board

    print(f"[ANIMATE] Called - is_animating: {is_animating}, index: {current_move_index}, data_len: {len(current_episode_data)}")

    # Check if animation should end
    if not is_animating or current_move_index >= len(current_episode_data):
        print(f"[ANIMATE] Animation complete, resuming training")
        # Animation finished. Reset flags
        is_animating = False
        current_move_index = 0
        board[:] = 0 
        
        # Return container that resumes training polling
        response = render(request, "game/stats_display_only.html", {
            "stats": training_state,
            "status": f"Episode {training_state['episode']} complete. Resuming training...",
        })
        return response
    
    # --- Apply the next move ---
    state, action, player, reward = current_episode_data[current_move_index]
    
    # Debug: Check state shape
    print(f"[ANIMATE] State type: {type(state)}, shape: {state.shape if hasattr(state, 'shape') else 'N/A'}")
    print(f"[ANIMATE] State content: {state}")
    
    # Use the state as-is (it's the board before the move)
    if isinstance(state, np.ndarray) and state.shape == (3, 3):
        board = state.copy()
    else:
        # State might be flattened or tuple
        board = np.array(state).reshape((3, 3))
    
    # Apply the action to show this move
    if action is not None:
        r, c = action
        board[r, c] = player 
        print(f"[ANIMATE] Move {current_move_index}: Player {player} -> ({r},{c})")
        print(f"[ANIMATE] Board after move:\n{board}")
    
    current_move_index += 1

    # Return just the board content
    context = {"board": board}
    return render(request, "game/board_content.html", context)


def reset_game(request):
    """Reset the game board for human play."""
    global board, current_player
    board[:] = 0
    current_player = 1
    return render(request, "game/board_content.html", {"board": board})


def get_training_stats(request):
    """API endpoint to get current training stats as JSON."""
    return JsonResponse({
        "episode": training_state["episode"],
        "wins": training_state["wins"],
        "losses": training_state["losses"],
        "draws": training_state["draws"],
        "epsilon": training_state["epsilon"],
        "is_animating": is_animating,
    })
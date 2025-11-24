"""
Play chess against the trained AI.

This script allows you to play interactively against a trained chess AI model.
Moves are entered in UCI format (e.g., 'e2e4' for moving a piece from e2 to e4).
"""

import torch
import numpy as np
import chess
import sys

from chess_ai import ChessInterface, ResNet


def display_board(board):
    """
    Display the chess board in ASCII format.
    
    Args:
        board: chess.Board instance
    """
    print("\n" + "=" * 33)
    print(board)
    print("=" * 33)


def get_ai_move(game, model, state):
    """
    Get the AI's move for the current state.
    
    Args:
        game: ChessInterface instance
        model: Trained neural network
        state: Current board state (FEN)
        
    Returns:
        UCI move string
    """
    # Encode state
    encoded_state = game.get_encoded_state(state)
    state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0).to(model.device)
    
    # Get policy from model
    with torch.no_grad():
        policy, value = model(state_tensor)
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
    
    # Mask illegal moves
    valid_moves = game.get_valid_moves_matrix(state)
    policy *= valid_moves
    
    # Normalize
    if np.sum(policy) > 0:
        policy /= np.sum(policy)
    else:
        # Fallback: uniform distribution over legal moves
        policy = valid_moves / np.sum(valid_moves)
    
    # Select move (greedy for best performance)
    action_idx = np.argmax(policy)
    move = game.actions[action_idx]
    
    return move, value.item()


def play_game(model_path="models/model_4.pt", num_res_blocks=8, num_hidden=64):
    """
    Play a game against the AI.
    
    Args:
        model_path: Path to trained model checkpoint
        num_res_blocks: Number of residual blocks (must match training)
        num_hidden: Number of hidden channels (must match training)
    """
    # Initialize game
    game = ChessInterface()
    board = chess.Board()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path}...")
    print(f"Using device: {device}")
    
    model = ResNet(game, num_res_blocks, num_hidden, device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully!\n")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train a model first using train.py")
        sys.exit(1)
    
    # Game setup
    print("Chess AI - Interactive Play")
    print("=" * 60)
    print("You play as White. Enter moves in UCI format (e.g., 'e2e4')")
    print("Type 'quit' to exit, 'moves' to see legal moves")
    print("=" * 60)
    
    state = game.get_initial_state()
    display_board(board)
    
    # Main game loop
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # Human move
            valid_moves = game.get_valid_moves(state)
            
            while True:
                move_input = input("\nYour move: ").strip().lower()
                
                if move_input == 'quit':
                    print("Thanks for playing!")
                    return
                elif move_input == 'moves':
                    print("\nLegal moves:", ", ".join(sorted(valid_moves)))
                    continue
                elif move_input in valid_moves:
                    action = move_input
                    break
                else:
                    print(f"Invalid move '{move_input}'. Type 'moves' to see legal moves.")
            
            # Apply human move
            state = game.get_next_state(state, action)
            board.set_fen(state)
            display_board(board)
            
        else:
            # AI move
            print("\nAI is thinking...")
            ai_move, position_value = get_ai_move(game, model, state)
            
            print(f"AI plays: {ai_move}")
            print(f"Position evaluation: {position_value:.3f} (AI perspective)")
            
            # Apply AI move
            state = game.get_next_state(state, ai_move)
            board.set_fen(state)
            display_board(board)
    
    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    
    if board.is_checkmate():
        winner = "Black (AI)" if board.turn == chess.WHITE else "White (You)"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate - Draw")
    elif board.is_insufficient_material():
        print("Draw - Insufficient material")
    elif board.is_seventyfive_moves():
        print("Draw - 75 move rule")
    elif board.is_fivefold_repetition():
        print("Draw - Fivefold repetition")
    else:
        print("Draw")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Play chess against the trained AI")
    parser.add_argument("--model", type=str, default="models/model_4.pt",
                       help="Path to model checkpoint (default: models/model_4.pt)")
    parser.add_argument("--res-blocks", type=int, default=8,
                       help="Number of residual blocks (default: 8)")
    parser.add_argument("--hidden", type=int, default=64,
                       help="Number of hidden channels (default: 64)")
    
    args = parser.parse_args()
    
    play_game(args.model, args.res_blocks, args.hidden)


if __name__ == "__main__":
    main()

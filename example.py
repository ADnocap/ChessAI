"""
Quick example/test script for the chess AI.

This script demonstrates how to use the chess AI components
without requiring a trained model or full training setup.
"""

import numpy as np
import torch
import chess

from chess_ai import ChessInterface, ResNet, MCTS


def test_board_encoding():
    """Test board state encoding."""
    print("Testing board encoding...")
    print("=" * 60)
    
    game = ChessInterface()
    state = game.get_initial_state()
    
    print(f"Initial state (FEN): {state}")
    print(f"\nAction size: {game.action_size}")
    
    # Encode state
    encoded = game.get_encoded_state(state)
    print(f"Encoded state shape: {encoded.shape}")
    print(f"Expected: (13, 8, 8) - 13 channels for pieces + empty squares")
    
    # Test valid moves
    valid_moves = game.get_valid_moves(state)
    print(f"\nNumber of legal moves from start: {len(valid_moves)}")
    print(f"First 5 moves: {valid_moves[:5]}")
    
    print("\n✓ Board encoding test passed!")


def test_model_forward():
    """Test model forward pass."""
    print("\nTesting model forward pass...")
    print("=" * 60)
    
    game = ChessInterface()
    device = torch.device("cpu")
    
    # Create small model for testing
    model = ResNet(game, num_resBlocks=2, num_hidden=16, device=device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    state = game.get_initial_state()
    encoded_state = game.get_encoded_state(state)
    state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        policy, value = model(state_tensor)
    
    print(f"\nPolicy output shape: {policy.shape}")
    print(f"Expected: (1, {game.action_size})")
    print(f"\nValue output shape: {value.shape}")
    print(f"Expected: (1, 1)")
    print(f"Value range: [{value.item():.3f}] (should be between -1 and 1)")
    
    # Apply softmax and check probabilities sum to 1
    policy_probs = torch.softmax(policy, dim=1)
    print(f"\nPolicy probabilities sum: {policy_probs.sum().item():.6f}")
    
    print("\n✓ Model forward pass test passed!")


def test_mcts_single_iteration():
    """Test a single MCTS iteration."""
    print("\nTesting MCTS (single iteration)...")
    print("=" * 60)
    
    game = ChessInterface()
    device = torch.device("cpu")
    
    model = ResNet(game, num_resBlocks=2, num_hidden=16, device=device)
    model.eval()
    
    args = {
        'C': 2,
        'num_searches': 10,  # Small number for quick test
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.002
    }
    
    mcts = MCTS(game, args, model)
    
    print("Running MCTS with 10 simulations...")
    state = game.get_initial_state()
    
    action_probs = mcts.search(state)
    
    print(f"\nAction probabilities shape: {action_probs.shape}")
    print(f"Sum of probabilities: {np.sum(action_probs):.6f}")
    
    # Find top 5 moves
    top_indices = np.argsort(action_probs)[-5:][::-1]
    print("\nTop 5 moves by MCTS:")
    for i, idx in enumerate(top_indices, 1):
        if action_probs[idx] > 0:
            print(f"  {i}. {game.actions[idx]}: {action_probs[idx]:.4f}")
    
    print("\n✓ MCTS test passed!")


def test_game_simulation():
    """Simulate a few moves of a game."""
    print("\nSimulating a short game...")
    print("=" * 60)
    
    game = ChessInterface()
    board = chess.Board()
    
    # Play a few moves
    moves = ['e2e4', 'e7e5', 'g1f3', 'b8c6']
    
    state = game.get_initial_state()
    
    for move in moves:
        print(f"\nPlaying: {move}")
        state = game.get_next_state(state, move)
        board.set_fen(state)
        
        value, is_terminal = game.get_value_and_terminated(state, move)
        print(f"Terminal: {is_terminal}, Value: {value}")
    
    print(f"\nFinal position:\n{board}")
    
    # Check valid moves from this position
    valid_moves = game.get_valid_moves(state)
    print(f"\nLegal moves from this position: {len(valid_moves)}")
    
    print("\n✓ Game simulation test passed!")


def main():
    """Run all tests."""
    print("\nChess AI - Component Tests")
    print("=" * 60)
    print("This script tests the basic functionality of the chess AI")
    print("without requiring a trained model.\n")
    
    try:
        test_board_encoding()
        test_model_forward()
        test_mcts_single_iteration()
        test_game_simulation()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run 'python generate_moves.py' to create the move encoding")
        print("2. Run 'python train.py' to train the model")
        print("3. Run 'python play.py' to play against the trained AI")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Train the chess AI using self-play and AlphaZero algorithm.

This script trains a ResNet-based neural network to play chess through
self-play reinforcement learning. The model learns by playing games against
itself and improving based on the outcomes.
"""

import torch
import os

from chess_ai.alphazero AlphaZeroParallel
from chess_ai.game import ChessInterface
from chess_ai.model import ResNet


def main():
    """Main training function."""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize game interface
    print("Initializing chess interface...")
    game = ChessInterface()
    
    # Set up device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model configuration
    num_res_blocks = 8
    num_hidden = 64
    
    print(f"Creating model with {num_res_blocks} residual blocks and {num_hidden} hidden channels...")
    model = ResNet(game, num_res_blocks, num_hidden, device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # Training parameters
    args = {
        'C': 2,                          # UCB exploration constant
        'num_searches': 100,             # Number of MCTS simulations per move
        'num_iterations': 5,             # Number of training iterations
        'num_selfPlay_iterations': 10,   # Number of self-play games per iteration
        'num_parallel_games': 2,         # Number of parallel games
        'num_epochs': 10,                # Training epochs per iteration
        'batch_size': 128,               # Training batch size
        'temperature': 1.25,             # Move selection temperature
        'dirichlet_epsilon': 0.25,       # Dirichlet noise weight
        'dirichlet_alpha': 0.002         # Dirichlet noise concentration
    }
    
    print("\nTraining configuration:")
    print("=" * 60)
    for key, value in args.items():
        print(f"{key:25s}: {value}")
    print("=" * 60)
    
    # Initialize AlphaZero trainer
    print("\nInitializing AlphaZero trainer...")
    alphazero = AlphaZeroParallel(model, optimizer, game, args)
    
    # Start training
    print("\nStarting training...")
    print("This may take a long time. Models will be saved after each iteration.")
    print("=" * 60)
    
    try:
        alphazero.learn()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving current model...")
        torch.save(model.state_dict(), "models/model_interrupted.pt")
        torch.save(optimizer.state_dict(), "models/optimizer_interrupted.pt")
        print("Model saved as model_interrupted.pt")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

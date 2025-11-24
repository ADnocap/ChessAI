# Chess AI with AlphaZero

A self-learning chess AI that combines reinforcement learning and deep neural networks to play chess. This project implements a chess engine using Monte Carlo Tree Search (MCTS) and a ResNet-based policy/value network, inspired by DeepMind's AlphaZero algorithm.

## Features

- **Self-Play Training**: The AI learns by playing games against itself
- **Monte Carlo Tree Search (MCTS)**: Efficient game tree exploration with parallel simulation support
- **Deep Residual Network**: Policy and value heads for move prediction and position evaluation
- **Complete Chess Implementation**: Full chess rules including castling, en passant, and promotion
- **Interactive Play**: Play against the trained AI through a simple interface

## Architecture

The AI consists of several key components:

1. **ResNet Model**: A convolutional neural network with residual blocks that outputs:

   - **Policy Head**: Probability distribution over all possible moves
   - **Value Head**: Position evaluation (-1 to 1 scale)

2. **Monte Carlo Tree Search**: Explores the game tree by balancing exploration and exploitation using the Upper Confidence Bound (UCB) formula

3. **Self-Play Training**: Generates training data through games played by the AI against itself, with continuous model improvement

## Quick Start

### Generate Move Matrix

First, generate the move encoding matrix (only needs to be done once):

```bash
python generate_moves.py
```

This creates a `moves.npy` file containing all possible chess moves encoded for the neural network.

### Train the Model

Train the AI using self-play:

```bash
python train.py
```

Training parameters can be adjusted in `train.py` including:

- Number of training iterations
- MCTS search depth
- Batch size
- Number of parallel games

### Play Against the AI

Once you have a trained model:

```bash
python play.py
```

## How It Works

### Board Representation

The chess board is encoded as 13 layers of 8×8 matrices:

- 12 layers for each piece type (6 pieces × 2 colors)
- 1 layer for empty squares

This representation allows the CNN to efficiently process spatial relationships on the board.

### Training Process

1. **Self-Play**: The model plays games against itself, using MCTS to select moves
2. **Data Collection**: Board states, move probabilities, and game outcomes are recorded
3. **Training**: The neural network is updated using:
   - Cross-entropy loss for the policy head (move prediction)
   - Mean squared error for the value head (position evaluation)
4. **Iteration**: The process repeats, with the model improving over time

### Monte Carlo Tree Search

MCTS explores possible moves by:

1. **Selection**: Traverse the tree using UCB to balance exploration/exploitation
2. **Expansion**: Add new nodes for unvisited positions
3. **Simulation**: Use the neural network to evaluate positions
4. **Backpropagation**: Update node statistics based on simulation results

## Training Configuration

Default hyperparameters (adjustable in `train.py`):

```python
{
    'C': 2,                          # UCB exploration constant
    'num_searches': 100,             # MCTS simulations per move
    'num_iterations': 5,             # Training iterations
    'num_selfPlay_iterations': 2,    # Self-play games per iteration
    'num_parallel_games': 2,         # Parallel game threads
    'num_epochs': 10,                # Training epochs per iteration
    'batch_size': 128,               # Training batch size
    'temperature': 1.25,             # Move selection temperature
    'dirichlet_epsilon': 0.25,       # Dirichlet noise weight
    'dirichlet_alpha': 0.002         # Dirichlet noise parameter
}
```

## Model Performance

The strength of the model depends heavily on:

- Number of training iterations
- MCTS search depth during both training and play
- Available computational resources
- Network depth (number of residual blocks)

Note: Training a strong chess AI requires significant computational resources. The provided configuration is designed for experimentation and may not reach expert-level play without extensive training.

## Computational Requirements

- **GPU**: Highly recommended for training (CUDA-compatible)
- **RAM**: At least 8GB
- **Storage**: Models are relatively small (~50MB each)
- **Training Time**: Varies significantly based on hardware
  - Each self-play game can take 20-40 minutes on CPU
  - GPU acceleration provides 10-100× speedup

## Extensions and Improvements

Possible enhancements to explore:

- Increase network depth (more residual blocks)
- Implement supervised learning from expert games
- Add opening book integration
- Implement endgame tablebases
- Optimize MCTS parallelization
- Add evaluation metrics and ELO rating estimation

## Limitations

- Requires substantial computational resources for strong play
- Training time is significant without GPU acceleration
- May not reach grandmaster level without extensive training
- MCTS depth limited by computational constraints

## References

This project draws inspiration from:

- [AlphaZero: Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero: Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## License

MIT License - feel free to use this project for learning and experimentation.

## Contributing

Contributions are welcome! Areas for improvement:

- Training optimization
- Better MCTS implementation
- Additional features (analysis, puzzles)
- Documentation improvements
- Performance benchmarking

## Acknowledgments

Built using:

- PyTorch for deep learning
- python-chess for chess game logic
- NumPy for numerical operations

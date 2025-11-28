"""
Chess AI with AlphaZero
A self-learning chess engine using deep reinforcement learning
"""

from game import ChessInterface
from model import ResNet, ResBlock
from mcts import MCTS, MCTSParallel, Node
from alphazero import AlphaZero, AlphaZeroParallel

__version__ = "1.0.0"
__all__ = [
    "ChessInterface",
    "ResNet",
    "ResBlock",
    "MCTS",
    "MCTSParallel",
    "Node",
    "AlphaZero",
    "AlphaZeroParallel",
]

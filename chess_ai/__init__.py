"""
Chess AI with AlphaZero
A self-learning chess engine using deep reinforcement learning
"""

from chess_ai.game import ChessInterface
from chess_ai.model import ResNet, ResBlock
from chess_ai.mcts import MCTS, MCTSParallel, Node
from chess_ai.alphazero import AlphaZero, AlphaZeroParallel

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

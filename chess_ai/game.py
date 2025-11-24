"""
Chess game interface and board representation.
"""

import numpy as np
import chess


class ChessInterface:
    """
    Interface for chess game management and board state encoding.
    
    This class handles all chess-specific logic including move generation,
    state transitions, and board encoding for neural network input.
    """
    
    def __init__(self, moves_file='moves.npy'):
        """
        Initialize the chess interface.
        
        Args:
            moves_file: Path to the numpy file containing move encodings
        """
        self.row_count = 8
        self.column_count = 8
        self.actions = np.load(moves_file)
        self.action_size = len(self.actions)

    def get_initial_state(self):
        """
        Get the starting position of a chess game.
        
        Returns:
            FEN string representing the initial board state
        """
        return chess.Board().fen()

    def get_next_state(self, state, action):
        """
        Apply a move to the current state.
        
        Args:
            state: Current board state (FEN string)
            action: Move in UCI format (e.g., 'e2e4')
            
        Returns:
            New board state after applying the move
        """
        board = chess.Board()
        board.set_fen(state)
        move = chess.Move.from_uci(action)
        board.push(move)
        return board.fen()

    def get_valid_moves(self, state):
        """
        Get all legal moves from the current position.
        
        Args:
            state: Current board state (FEN string)
            
        Returns:
            List of legal moves in UCI format
        """
        board = chess.Board()
        board.set_fen(state)
        return [move.uci() for move in board.legal_moves]

    def get_valid_moves_matrix(self, state):
        """
        Get a binary matrix indicating which moves are legal.
        
        Args:
            state: Current board state (FEN string)
            
        Returns:
            Binary numpy array of shape (action_size,) where 1 indicates legal move
        """
        valid_moves = self.get_valid_moves(state)
        valid_move_matrix = np.zeros(self.action_size)
        
        for i, move in enumerate(self.actions):
            if move in valid_moves:
                valid_move_matrix[i] = 1
                
        return valid_move_matrix.astype(np.float32)

    def check_win(self, state, action):
        """
        Check if the given action results in checkmate.
        
        Args:
            state: Current board state
            action: Move to check
            
        Returns:
            True if the position is checkmate, False otherwise
        """
        if action is None:
            return False
        board = chess.Board()
        board.set_fen(state)
        return board.is_checkmate()

    def get_value_and_terminated(self, state, action):
        """
        Evaluate the game state and check if game is over.
        
        Args:
            state: Current board state
            action: Last action taken
            
        Returns:
            Tuple of (value, is_terminal) where value is 1 for win, 0 for draw/ongoing
        """
        board = chess.Board()
        board.set_fen(state)
        
        if self.check_win(state, action):
            return 1, True
        if board.outcome() is not None:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        """Switch player perspective."""
        return -player

    def get_opponent_value(self, value):
        """Flip the value for the opponent's perspective."""
        return -value

    def get_encoded_state(self, state):
        """
        Encode the board state as a tensor for neural network input.
        
        The board is encoded as 13 channels of 8x8 matrices:
        - 12 channels for each piece type (6 pieces × 2 colors)
        - 1 channel for empty squares
        
        Args:
            state: Board state (FEN string) or array of states
            
        Returns:
            Numpy array of shape (13, 8, 8) or (batch_size, 13, 8, 8)
        """
        return_matrices = []

        # Handle batch of states
        if not isinstance(state, (str, np.str_)):
            for s in state:
                encoded = self._encode_single_state(s)
                return_matrices.append(encoded)
        else:
            # Single state
            return_matrices = self._encode_single_state(state)
            
        return np.array(return_matrices).astype(np.float32)

    def _encode_single_state(self, state):
        """
        Encode a single board state.
        
        Args:
            state: FEN string
            
        Returns:
            Array of shape (13, 8, 8)
        """
        board = chess.Board(state)
        pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
        matrices = []

        # Create a matrix for each piece type
        for piece in pieces:
            matrix = np.zeros((8, 8))
            for i in range(64):
                if str(board.piece_at(i)) == piece:
                    matrix[i // 8, i % 8] = 1
            matrices.append(matrix)

        # Add matrix for empty squares
        free_squares = np.zeros((8, 8))
        for i in range(64):
            if board.piece_at(i) is None:
                free_squares[i // 8, i % 8] = 1
        matrices.append(free_squares)

        return np.array(matrices)

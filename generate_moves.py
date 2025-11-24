"""
Generate the move encoding matrix for the chess AI.

This script creates a numpy array containing all possible chess moves
encoded in UCI format. This includes:
- All possible queen moves (sliding moves)
- All possible knight moves
- Pawn promotions

The resulting moves.npy file is used by the neural network to map
between move indices and actual chess moves.
"""

import numpy as np
import chess


def generate_moves(piece):
    """
    Generate all possible moves for a piece type from every square.
    
    Args:
        piece: chess.Piece instance
        
    Returns:
        List of lists, where each inner list contains UCI move strings
        for that piece from a specific square
    """
    board = chess.Board(None)  # Create an empty chess board
    moves = []

    for square in chess.SQUARES:
        board.set_piece_at(square, piece)
        moves.append([str(move) for move in board.legal_moves])
        board.remove_piece_at(square)

    return moves


def get_move_matrix():
    """
    Create the complete move encoding matrix.
    
    Generates all possible moves by:
    1. Computing queen moves (covers all sliding moves: horizontal, vertical, diagonal)
    2. Computing knight moves
    3. Adding pawn promotion moves explicitly
    
    Returns:
        numpy array of UCI move strings
    """
    move_list = []
    move_q = {}
    move_k = {}

    # Generate queen moves
    print("Generating queen moves...")
    queen_moves = generate_moves(chess.Piece(chess.QUEEN, chess.WHITE))
    for square, moves in zip(chess.SQUARE_NAMES, queen_moves):
        move_q[square] = moves

    # Generate knight moves
    print("Generating knight moves...")
    knight_moves = generate_moves(chess.Piece(chess.KNIGHT, chess.WHITE))
    for square, moves in zip(chess.SQUARE_NAMES, knight_moves):
        move_k[square] = moves

    # Combine queen and knight moves
    print("Combining moves...")
    for square in chess.SQUARE_NAMES:
        for move in move_q[square]:
            move_list.append(move)
        for move in move_k[square]:
            move_list.append(move)

    # Add pawn promotion moves (queen promotion only)
    print("Adding promotion moves...")
    
    # White promotions (7th rank to 8th rank)
    white_promotions = [
        "a7a8q", "a7b8q",
        "b7b8q", "b7a8q", "b7c8q",
        "c7c8q", "c7b8q", "c7d8q",
        "d7d8q", "d7c8q", "d7e8q",
        "e7e8q", "e7d8q", "e7f8q",
        "f7f8q", "f7e8q", "f7g8q",
        "g7g8q", "g7f8q", "g7h8q",
        "h7h8q", "h7g8q"
    ]
    
    # Black promotions (2nd rank to 1st rank)
    black_promotions = [
        "a2a1q", "a2b1q",
        "b2b1q", "b2a1q", "b2c1q",
        "c2c1q", "c2b1q", "c2d1q",
        "d2d1q", "d2c1q", "d2e1q",
        "e2e1q", "e2d1q", "e2f1q",
        "f2f1q", "f2e1q", "f2g1q",
        "g2g1q", "g2f1q", "g2h1q",
        "h2h1q", "h2g1q"
    ]
    
    move_list.extend(white_promotions)
    move_list.extend(black_promotions)

    # Convert to numpy array and save
    move_array = np.array(move_list)
    np.save('moves.npy', move_array)
    
    print(f"\nTotal moves: {len(move_array)}")
    print("Saved to moves.npy")
    
    return move_array


if __name__ == "__main__":
    print("Generating chess move encoding matrix...")
    print("=" * 60)
    moves = get_move_matrix()
    print("=" * 60)
    print("Done!")

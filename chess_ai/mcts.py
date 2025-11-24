"""
Monte Carlo Tree Search implementation.
Supports both sequential and parallel MCTS for efficient game tree exploration.
"""

import numpy as np
import math
import torch


class Node:
    """
    Node in the Monte Carlo search tree.
    
    Each node represents a game state and contains statistics for action selection.
    """
    
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        """
        Initialize a search tree node.
        
        Args:
            game: ChessInterface instance
            args: Dictionary of MCTS parameters
            state: Board state (FEN string)
            parent: Parent node (None for root)
            action_taken: Action that led to this state
            prior: Prior probability from policy network
            visit_count: Number of visits to this node
        """
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        """Check if this node has been expanded."""
        return len(self.children) > 0

    def select(self):
        """
        Select the best child using Upper Confidence Bound (UCB).
        
        Returns:
            Child node with highest UCB value
        """
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child

    def get_ucb(self, child):
        """
        Calculate Upper Confidence Bound for a child node.
        
        Balances exploitation (q_value) and exploration (prior * sqrt term).
        
        Args:
            child: Child node to evaluate
            
        Returns:
            UCB value
        """
        if child.visit_count == 0:
            q_value = 0
        else:
            # Normalize value to [0, 1] range
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
            
        exploration = self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
        return q_value + exploration

    def expand(self, policy):
        """
        Expand this node by creating children for all legal moves.
        
        Args:
            policy: Probability distribution over actions
        """
        for action, prob in enumerate(policy):
            if prob > 0:
                move = self.game.actions[action]
                child_state = self.game.get_next_state(self.state, move)
                child = Node(self.game, self.args, child_state, self, move, prob)
                self.children.append(child)

    def backpropagate(self, value):
        """
        Update statistics along the path to root.
        
        Args:
            value: Value to backpropagate (from current player's perspective)
        """
        self.value_sum += value
        self.visit_count += 1
        
        # Flip value for opponent
        value = self.game.get_opponent_value(value)
        
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    """
    Monte Carlo Tree Search for single-threaded game tree exploration.
    """
    
    def __init__(self, game, args, model):
        """
        Initialize MCTS.
        
        Args:
            game: ChessInterface instance
            args: Dictionary of MCTS parameters
            model: Neural network model
        """
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        """
        Perform MCTS from the given state.
        
        Args:
            state: Current board state
            
        Returns:
            Action probability distribution based on visit counts
        """
        # Create root node
        root = Node(self.game, self.args, state, visit_count=1)
        
        # Get initial policy from neural network
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        
        # Add Dirichlet noise for exploration
        policy = ((1 - self.args['dirichlet_epsilon']) * policy + 
                  self.args['dirichlet_epsilon'] * 
                  np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size))
        
        # Mask invalid moves
        valid_moves = self.game.get_valid_moves_matrix(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        
        # Expand root
        root.expand(policy)
        
        # Perform MCTS iterations
        for _ in range(self.args['num_searches']):
            # Selection: traverse to leaf node
            node = root
            while node.is_fully_expanded():
                node = node.select()
            
            # Check if terminal
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                # Expansion: get policy and value from neural network
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state),
                                 device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                
                # Mask invalid moves
                valid_moves = self.game.get_valid_moves_matrix(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                node.expand(policy)
            
            # Backpropagation
            node.backpropagate(value)
        
        # Return action probabilities based on visit counts
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[np.where(self.game.actions == child.action_taken)] = child.visit_count
        action_probs /= np.sum(action_probs)
        
        return action_probs


class MCTSParallel:
    """
    Parallel Monte Carlo Tree Search for efficient batch processing.
    
    Processes multiple game states simultaneously for improved throughput.
    """
    
    def __init__(self, game, args, model):
        """
        Initialize parallel MCTS.
        
        Args:
            game: ChessInterface instance
            args: Dictionary of MCTS parameters
            model: Neural network model
        """
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        """
        Perform parallel MCTS on multiple states.
        
        Args:
            states: List of board states
            spGames: List of self-play game objects
            
        The function modifies spGames in place, updating their root nodes
        with MCTS results.
        """
        # Get policies for all states in parallel
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        
        # Add Dirichlet noise for exploration
        policy = ((1 - self.args['dirichlet_epsilon']) * policy + 
                  self.args['dirichlet_epsilon'] * 
                  np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, 
                                     size=policy.shape[0]))

        # Initialize root nodes for each game
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves_matrix(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)

        # Perform MCTS searches
        for _ in range(self.args['num_searches']):
            # Selection phase for all games
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            # Batch expansion for non-terminal nodes
            expandable_spGames = [idx for idx in range(len(spGames)) if spGames[idx].node is not None]

            if len(expandable_spGames) > 0:
                # Evaluate all expandable states in parallel
                states_to_expand = np.stack([spGames[idx].node.state for idx in expandable_spGames])

                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states_to_expand), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

                # Expand and backpropagate for each game
                for i, idx in enumerate(expandable_spGames):
                    node = spGames[idx].node
                    spg_policy, spg_value = policy[i], value[i]

                    valid_moves = self.game.get_valid_moves_matrix(node.state)
                    spg_policy *= valid_moves
                    spg_policy /= np.sum(spg_policy)

                    node.expand(spg_policy)
                    node.backpropagate(spg_value)

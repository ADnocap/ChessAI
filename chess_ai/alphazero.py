"""
AlphaZero training algorithm.
Implements self-play and training loop for the chess AI.
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
import chess
from tqdm import trange

from chess_ai.mcts import MCTS, MCTSParallel


class SPG:
    """
    Self-Play Game container.
    
    Stores the state and memory for a single self-play game.
    """
    
    def __init__(self, game):
        """
        Initialize a self-play game.
        
        Args:
            game: ChessInterface instance
        """
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None


class AlphaZero:
    """
    AlphaZero training algorithm (sequential version).
    
    Implements self-play data generation and neural network training.
    """
    
    def __init__(self, model, optimizer, game, args):
        """
        Initialize AlphaZero trainer.
        
        Args:
            model: Neural network model
            optimizer: PyTorch optimizer
            game: ChessInterface instance
            args: Dictionary of training parameters
        """
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfplay(self):
        """
        Play one game through self-play.
        
        Returns:
            List of (state, action_probs, outcome) tuples for training
        """
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            # Get move probabilities from MCTS
            action_probs = self.mcts.search(state)
            memory.append((state, action_probs, player))

            # Sample move with temperature
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs = temperature_action_probs / np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            action = self.game.actions[action]

            # Apply move
            state = self.game.get_next_state(state, action)

            # Check if game is over
            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                # Process game history with final outcome
                return_memory = []
                for hist_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    return_memory.append((
                        self.game.get_encoded_state(hist_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return return_memory

            player = self.game.get_opponent(player)

    def train(self, memory):
        """
        Train the neural network on game data.
        
        Args:
            memory: List of (state, policy_target, value_target) tuples
        """
        random.shuffle(memory)
        
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory) - 1, batch_idx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            # Convert to tensors
            state = np.array(state)
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            # Forward pass
            out_policy, out_value = self.model(state)

            # Calculate losses
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        """
        Main training loop.
        
        Alternates between self-play data generation and neural network training.
        """
        for iteration in trange(self.args['num_iterations'], desc="Training iterations"):
            memory = []

            # Self-play phase
            self.model.eval()
            for _ in trange(self.args['num_selfPlay_iterations'], desc="Self-play games", leave=False):
                memory += self.selfplay()

            # Training phase
            self.model.train()
            for _ in trange(self.args['num_epochs'], desc="Training epochs", leave=False):
                self.train(memory)

            # Save checkpoint
            torch.save(self.model.state_dict(), f"models/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"models/optimizer_{iteration}.pt")


class AlphaZeroParallel:
    """
    AlphaZero training algorithm (parallel version).
    
    Processes multiple self-play games simultaneously for improved efficiency.
    """
    
    def __init__(self, model, optimizer, game, args):
        """
        Initialize parallel AlphaZero trainer.
        
        Args:
            model: Neural network model
            optimizer: PyTorch optimizer
            game: ChessInterface instance
            args: Dictionary of training parameters
        """
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfplay(self):
        """
        Play multiple games in parallel through self-play.
        
        Returns:
            List of (state, action_probs, outcome) tuples for training
        """
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for _ in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            # Get all current states
            states = np.stack([spg.state for spg in spGames])

            # Run MCTS for all games in parallel
            self.mcts.search(states, spGames)

            # Process each game
            for i in range(len(spGames) - 1, -1, -1):
                spg = spGames[i]

                # Extract action probabilities from visit counts
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[np.where(self.game.actions == child.action_taken)] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                # Sample move with temperature
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs = temperature_action_probs / np.sum(temperature_action_probs)
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                action = self.game.actions[action]

                # Apply move
                spg.state = self.game.get_next_state(spg.state, action)

                # Check if game is over
                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    # Process this game's history
                    for hist_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    # Remove completed game
                    del spGames[i]

            # Switch players
            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
        """
        Train the neural network on game data.
        
        Args:
            memory: List of (state, policy_target, value_target) tuples
        """
        random.shuffle(memory)
        
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory) - 1, batch_idx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            # Convert to tensors
            state = np.array(state)
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            # Forward pass
            out_policy, out_value = self.model(state)

            # Calculate losses
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        """
        Main training loop with parallel self-play.
        
        Alternates between parallel self-play data generation and neural network training.
        """
        for iteration in trange(self.args['num_iterations'], desc="Training iterations"):
            memory = []

            # Self-play phase (parallel)
            self.model.eval()
            num_batches = self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']
            for _ in trange(num_batches, desc="Self-play batches", leave=False):
                memory += self.selfplay()

            # Training phase
            self.model.train()
            for _ in trange(self.args['num_epochs'], desc="Training epochs", leave=False):
                self.train(memory)

            # Save checkpoint
            torch.save(self.model.state_dict(), f"models/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"models/optimizer_{iteration}.pt")
            
            print(f"Iteration {iteration}: Generated {len(memory)} training examples")

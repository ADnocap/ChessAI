"""
Neural network architecture for the chess AI.
Implements a ResNet-based model with policy and value heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual block for deep network training.
    
    Uses skip connections to allow gradients to flow through deep networks.
    """
    
    def __init__(self, num_hidden):
        """
        Initialize residual block.
        
        Args:
            num_hidden: Number of hidden channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after residual block
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    """
    ResNet-based neural network for chess position evaluation.
    
    The network has two heads:
    - Policy head: Outputs probability distribution over moves
    - Value head: Outputs position evaluation (-1 to 1)
    """
    
    def __init__(self, game, num_resBlocks, num_hidden, device):
        """
        Initialize the ResNet model.
        
        Args:
            game: ChessInterface instance
            num_resBlocks: Number of residual blocks
            num_hidden: Number of hidden channels
            device: torch device (cpu or cuda)
        """
        super().__init__()
        
        self.device = device
        
        # Initial convolution
        self.startBlock = nn.Sequential(
            nn.Conv2d(13, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        # Residual tower
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )
        
        # Policy head
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        
        # Value head
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 13, 8, 8)
            
        Returns:
            Tuple of (policy, value) where:
            - policy: logits over actions (batch_size, action_size)
            - value: position evaluation (batch_size, 1)
        """
        x = self.startBlock(x)
        
        for resBlock in self.backBone:
            x = resBlock(x)
            
        policy = self.policyHead(x)
        value = self.valueHead(x)
        
        return policy, value

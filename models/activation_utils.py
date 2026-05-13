

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_activation(activation_type: str = 'relu'):
   
    if activation_type.lower() == 'relu':
        return nn.ReLU()
    elif activation_type.lower() == 'gelu':
        return nn.GELU()
    elif activation_type.lower() == 'swish' or activation_type.lower() == 'silu':
        return nn.SiLU()  # Swish is the same as SiLU
    elif activation_type.lower() == 'glu':
        
        return None
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


def build_mlp_with_activation(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    activation_type: str = 'relu',
    use_layer_norm: bool = False,
    dropout: float = 0.0,
    output_activation: Optional[str] = None
):
   
    if num_layers < 2:
        raise ValueError("num_layers must be at least 2")
    
    layers = []
    
  
    if activation_type.lower() == 'glu':

        layers.append(nn.Linear(input_dim, hidden_dim * 2))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim * 2))
        layers.append(nn.GLU(dim=-1))
    else:
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        activation = get_activation(activation_type)
        if activation is not None:
            layers.append(activation)
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    
    
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        if activation_type.lower() == 'glu':
            layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
            layers.append(nn.GLU(dim=-1))
        else:
            activation = get_activation(activation_type)
            if activation is not None:
                layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    if output_activation is not None:
        if output_activation.lower() == 'relu':
            layers.append(nn.ReLU())
        elif output_activation.lower() == 'softplus':
            layers.append(nn.Softplus())
        elif output_activation.lower() == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation.lower() == 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")
    
    return nn.Sequential(*layers)


class GLUMLP(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        
        self.layers.append(nn.Linear(input_dim, hidden_dim * 2))
        
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
        
        
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.glu(x, dim=-1)
        x = self.layers[-1](x)
        return x


import torch
import torch.nn as nn
from .routing_layer import RoutingLayer


class AutoNN(nn.Module):

    def __init__(self, num_bots, hidden_dim):
        super(AutoNN, self).__init__()

        self.num_bots = num_bots

        self.bot_layer = nn.Linear(num_bots, hidden_dim)

        self.routing = RoutingLayer(hidden_dim, num_bots)

    def forward(self, x):

        x = torch.relu(self.bot_layer(x))

        routing_probs = self.routing(x)

        return routing_probs

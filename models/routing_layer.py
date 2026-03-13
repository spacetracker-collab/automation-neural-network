import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingLayer(nn.Module):
    """
    Routing layer for Automation Neural Network (AutoNN)

    Implements softmax-based task routing between bots.
    """

    def __init__(self, input_dim, num_bots):
        super(RoutingLayer, self).__init__()

        # linear transformation to produce routing scores
        self.routing_weights = nn.Linear(input_dim, num_bots)

    def forward(self, x):

        # compute routing scores
        scores = self.routing_weights(x)

        # convert scores to probabilities
        routing_probs = F.softmax(scores, dim=-1)

        return routing_probs

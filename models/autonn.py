import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoNN(nn.Module):

    def __init__(self, num_bots, hidden_dim):
        super(AutoNN, self).__init__()

        self.num_bots = num_bots

        # workflow weight matrix
        self.weights = nn.Parameter(torch.randn(num_bots, num_bots))

        # bot biases
        self.bias = nn.Parameter(torch.zeros(num_bots))

        # bot processing layer
        self.bot_layer = nn.Linear(num_bots, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, num_bots)

    def forward(self, x):

        # propagate across automation graph
        x = torch.matmul(x, self.weights)

        x = x + self.bias

        x = torch.relu(self.bot_layer(x))

        x = self.output_layer(x)

        # softmax routing
        routing_prob = F.softmax(x, dim=-1)

        return routing_prob

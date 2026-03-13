import torch
from models.autonn import AutoNN

num_bots = 10
hidden_dim = 32

model = AutoNN(num_bots, hidden_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(100):

    x = torch.randn(1, num_bots)

    target = torch.randint(0, num_bots, (1,))

    output = model(x)

    loss = loss_fn(output, target)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("Epoch:", epoch, "Loss:", loss.item())

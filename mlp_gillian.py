import torch
import numpy as np
from torch import nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(4,5, bias = True)
        self.layer2 = nn.Linear(5, 4, bias=True)
        self.layer3 = nn.Linear(4, 1, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)

        return x

if __name__ == '__main__':
    samples = np.random.randn(5000, 4)
    ground_truth = 1/(1+np.exp(-samples.sum(1)))
    # ground_truth = 0.5*samples.sum(1) + 1

    model = MLP()

    optimizer = SGD(model.parameters(), lr = 0.001)

    loss = []

    # training loop
    for epoch in range(10000):
        input = torch.tensor(samples, dtype=torch.float32)
        gt = torch.tensor(ground_truth, dtype=torch.float32).unsqueeze(1)

        output = model(input)
        optimizer.zero_grad()
        loss_val = nn.functional.l1_loss(output, gt)
        loss_val.backward()
        optimizer.step()
        loss.append(loss_val.item())

    print(loss[-1])
    plt.plot(loss)
    plt.show()
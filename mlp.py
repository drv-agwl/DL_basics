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
        self.layer1 = nn.Linear(4, 1, bias=True)

    def forward(self, input):
        x1 = self.layer1(input) # (1, 4) -> (1, 1)

        return x1


if __name__ == '__main__':
    samples = np.random.randn(5000, 4)
    # ground_truth = 1/(1+np.exp(-samples.sum(1)))
    ground_truth = 0.5*samples.sum(1) + 1

    model = MLP().to(device='mps')

    optimizer = SGD(model.parameters(), lr=0.01)
    lr_scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

    loss_all_samples = []

    pbar = tqdm(range(10000))
    for epoch in range(10000):
        x = torch.tensor(samples, dtype=torch.float32).to(device='mps').unsqueeze(0)
        gt = torch.tensor(ground_truth, dtype=torch.float32).to(device='mps').unsqueeze(1)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.L1Loss()(output, gt)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_all_samples.append(loss.item())
        print(f"Epoch {epoch}: Loss: {loss.item()}")
        pbar.update(1)

    plt.plot(loss_all_samples)
    plt.show()

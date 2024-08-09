from torch import nn

class PsrNeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.dead_simple_stack = nn.Sequential(
      nn.Linear(6, 9),
      nn.ReLU(),
      nn.Linear(9, 3),
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.dead_simple_stack(x)
    return logits

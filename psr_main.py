from torch.utils.data import DataLoader
from torch import nn
import torch

from psr_dataset import PsrDataset
from psr_network import PsrNeuralNetwork

device = "cpu"

training_data = PsrDataset(3000)
test_data = PsrDataset(50)

train_dataloader = DataLoader(training_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)

model = PsrNeuralNetwork().to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    pred = model(X)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss, current = loss.item(), (batch + 1) * len(X)
    print(f"loss: {loss:>7f} {current:>5d}/{size:>5d}")

def decode_input(data):
  if data[0] > 0:
    return "P"
  elif data[1] > 0:
    return "S"
  elif data[2] > 0:
    return "R"
  else:
    return "Unknown"

def decode_pred(data):
  a_wins = data[0]
  draw = data[1]
  b_wins = data[2]

  if a_wins > draw and a_wins > b_wins:
    return ">"
  elif draw > a_wins and draw > b_wins:
    return "="
  elif b_wins > draw and b_wins > a_wins:
    return "<"
  else:
    return "Unknown"

def test(dataloader, model, loss_fn):
  model.eval()
  size = len(dataloader.dataset)
  correct = 0
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      pred_result = decode_pred(pred[0])
      should_be = decode_pred(y[0])
      if pred_result == should_be:
        correct += 1
      print(
        f"{(decode_input(X[0][0]))} vs {(decode_input(X[0][1]))}" +
        f" pred result: {pred_result}" +
        f" should be: {should_be}",
      )
  print(f"correct: {correct}/{size}")

train(train_dataloader, model, loss_fn, optimizer)
test(test_dataloader, model, loss_fn)

for param in model.parameters():
  print(param.data)

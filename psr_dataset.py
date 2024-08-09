import torch
from torch.utils.data import Dataset

import random

class PsrDataset(Dataset):
  def __init__(self, length):
    self.length = length
    self.data = [
      { "playerA": "P", "playerB": "P", "result": "D" },
      { "playerA": "S", "playerB": "S", "result": "D" },
      { "playerA": "R", "playerB": "R", "result": "D" },

      { "playerA": "P", "playerB": "R", "result": "A" },
      { "playerA": "R", "playerB": "S", "result": "A" },
      { "playerA": "S", "playerB": "P", "result": "A" },

      { "playerA": "P", "playerB": "S", "result": "B" },
      { "playerA": "S", "playerB": "R", "result": "B" },
      { "playerA": "R", "playerB": "P", "result": "B" },
    ]

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    current_data = self.data[random.randint(0, len(self.data) - 1)]
    input_tensor_a = self.__get_input_tensor(current_data["playerA"])
    input_tensor_b = self.__get_input_tensor(current_data["playerB"])
    input_tensor = torch.tensor([
      input_tensor_a,
      input_tensor_b,
    ])
    label_tensor = torch.tensor(
      self.__get_label_tensor(current_data["result"]),
    )
    return input_tensor, label_tensor

  def __get_input_tensor(self, ipt: str):
    if ipt == "P":
      return [1.0, -1.0, -1.0]
    elif ipt == "S":
      return [-1.0, 1.0, -1.0]
    elif ipt == "R":
      return [-1.0, -1.0, 1.0]
    else:
      return [-1.0, -1.0, -1.0]

  def __get_label_tensor(self, res: str):
    if res == "D":
      return [-1.0, 1.0, -1.0]
    elif res == "A":
      return [1.0, -1.0, -1.0]
    elif res == "B":
      return [-1.0, -1.0, 1.0]
    else:
      return [-1.0, -1.0, -1.0]

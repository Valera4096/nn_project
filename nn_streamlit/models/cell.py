import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def decode(x):
    d = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}
    return d[x]

from torchvision.models import resnet152

model_cell = resnet152()
model_cell.fc = nn.Linear(2048, 4)

model_cell.load_state_dict(torch.load('models/weights_cell.pt'))



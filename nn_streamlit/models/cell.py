import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def decode(x):
    d = {0: 'EOSINOPHIL', 1: 'LYMPHOCYTE', 2: 'MONOCYTE', 3: 'NEUTROPHIL'}
    return d[x]


from torchvision.models import mobilenet_v3_small


model_ = mobilenet_v3_small()
model_.classifier[3] = nn.Linear(1024, 4)

model_.to(device)

model_.load_state_dict(torch.load('/home/valera/ds_bootcamp/nn_project/nn_streamlit/models/weights_cell_2.pt'))

model_cell = model_



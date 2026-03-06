import torch
from model import CNN   # your CNN class

model = CNN()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
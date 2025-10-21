import torch
from torch.utils.data import TensorDataset, DataLoader

data = torch.load("rat_vision_dataset.pt", map_location="cpu")
ds = TensorDataset(data["x"], data["y"])

loader = DataLoader(ds, batch_size=32, shuffle=True)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)
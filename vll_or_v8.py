import torch

model_path = r"C:\Users\USER\Documents\Latest_LPR\runs\detect\train16\weights\best.pt"
model = torch.load(model_path, map_location="cpu")

print(model)

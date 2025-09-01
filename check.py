import torch
ckpt = torch.load('runs/train/exp41/weights/best.pt', map_location='cpu')
print("Model nc:", ckpt['model'].nc)
print("Names:", ckpt['model'].names)
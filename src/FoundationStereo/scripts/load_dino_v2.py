import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
print('DINOv2 loaded!')
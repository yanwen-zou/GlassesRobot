import torch
from pytorch3d.ops import knn_points

# 构造一个点云 (batch=1, num_points=5, dim=3)
p1 = torch.rand(1, 5, 3).cuda()
p2 = torch.rand(1, 10, 3).cuda()

# 调用 knn 搜索
dist, idx, nn = knn_points(p1, p2, K=1)
print("KNN works, dist:", dist)

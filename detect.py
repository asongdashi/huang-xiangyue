import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def vertical_line_direction(feature_map):
    """
    输入: feature_map [1, H, W] 单通道特征图
    输出: 方向向量 (dx, dy) 单位向量
    """
    _, H, W = feature_map.shape

    # -------------------------
    # 1. 定义两个左右对称卷积核
    # -------------------------
    K1 = torch.tensor([[-1,0,1],
                       [-1,0,1],
                       [-1,0,1]], dtype=feature_map.dtype, device=feature_map.device).view(1,1,3,3)
    K2 = torch.tensor([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]], dtype=feature_map.dtype, device=feature_map.device).view(1,1,3,3)

    # -------------------------
    # 2. 卷积操作
    # -------------------------
    conv1_out = F.conv2d(feature_map.unsqueeze(0), K1, padding=1)
    conv2_out = F.conv2d(feature_map.unsqueeze(0), K2, padding=1)

    # -------------------------
    # 3. 计算重心
    # -------------------------
    def compute_centroid(conv_out):
        conv_out = conv_out[0,0]  # [H,W]
        # 水平重心
        x_weighted = torch.arange(W, device=conv_out.device).view(1,W) * conv_out
        x_center = x_weighted.sum(dim=1).sum() / (conv_out.sum() + 1e-6)
        # 垂直重心
        y_weighted = torch.arange(H, device=conv_out.device).view(H,1) * conv_out
        y_center = y_weighted.sum() / (conv_out.sum() + 1e-6)
        return x_center, y_center

    x1, y1 = compute_centroid(conv1_out)
    x2, y2 = compute_centroid(conv2_out)

    # -------------------------
    # 4. 计算方向向量
    # -------------------------
    dx = x2 - x1
    dy = y2 - y1
    vec = torch.tensor([dx, dy], device=feature_map.device)
    norm = torch.norm(vec)
    if norm < 1e-6:
        direction = torch.tensor([0.0, 0.0], device=feature_map.device)
    else:
        direction = vec / norm  # 单位向量
    return direction

H, W = 64, 64
feat = torch.zeros(1, H, W)

# -------------------------
# 模拟一条斜线，宽度为3
# y = 0.5*x + 10
# -------------------------
width = 3
for x in range(W):
    y = int(0.1 * x + 10)
    for w in range(-width//2, width//2+1):
        y_idx = y + w
        if 0 <= y_idx < H:
            feat[0, y_idx, x] = 1.0

# 可视化特征图
# plt.imshow(feat[0], cmap='gray')
# plt.title("斜线特征图")
# plt.show()

# -------------------------
# 调用 vertical_line_direction
# -------------------------
direction = vertical_line_direction(feat)
print("方向向量:", direction)

# =========================================
# 🌙 双月数据集 + Neural Network Training
# Decision Boundary + Live Loss Curve
# 中文 + English (中英文对照)
# =========================================

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.datasets import make_moons
from matplotlib import rcParams
import sys
# -------------------------------------------------
# 🈶 设置中文字体（黑体/思源黑体/微软雅黑）
# -------------------------------------------------
rcParams['font.sans-serif'] = ['SimHei']   # 设置字体为黑体（支持中文）

rcParams['axes.unicode_minus'] = False      # 解决负号显示问题

# -------------------------------------------------
# 1️⃣ 生成双月数据集 / Generate Double Moon Dataset
# -------------------------------------------------
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# -------------------------------------------------
# 2️⃣ 定义神经网络 / Define Neural Network
# -------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

net = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# -------------------------------------------------
# 3️⃣ 构造网格 / Create Grid for Decision Boundary
# -------------------------------------------------
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# -------------------------------------------------
# 4️⃣ 创建画布 / Create Subplots
# -------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("双月数据集神经网络训练动画 / Two-Moon Neural Network Training Animation", fontsize=14)

losses = []
accuracies = []

# -------------------------------------------------
# 5️⃣ 计算准确率函数 / Accuracy Function
# -------------------------------------------------
def accuracy(preds, labels):
    pred_classes = (preds > 0.5).float()
    return (pred_classes == labels).float().mean().item()

# -------------------------------------------------
# 6️⃣ 动画更新函数 / Animation Update Function
# -------------------------------------------------
def update(frame):
    # 每帧训练若干次 / Train multiple times per frame for faster convergence
    for _ in range(10):
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    acc = accuracy(outputs, y)
    accuracies.append(acc)

    # 计算预测结果 / Compute Predictions for Decision Boundary
    with torch.no_grad():
        preds = net(grid).reshape(xx.shape)

    # 左图：决策边界 / Decision Boundary
    ax1.clear()
    ax1.contourf(xx, yy, preds, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.6)
    ax1.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap="coolwarm", edgecolor='k')
    ax1.set_xlim(-2, 3)
    ax1.set_ylim(-1.5, 2)
    ax1.set_title(f"决策边界 / Decision Boundary (Epoch {frame*10})", fontsize=12)

    # UserWarning: Glyph 8322 (\N{SUBSCRIPT TWO}) missing from font(s)
    # ax1.set_xlabel("特征 x₁ / Feature x₁")
    # ax1.set_ylabel("特征 x₂ / Feature x₂")

    ax1.set_xlabel("特征 $x_1$ / Feature $x_1$")
    ax1.set_ylabel("特征 $x_2$ / Feature $x_2$")

    # 显示当前损失与准确率 / Show Loss and Accuracy
    ax1.text(-1.9, 1.7, f"损失 Loss: {loss.item():.3f}\n准确率 Accuracy: {acc*100:.1f}%",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # 右图：损失曲线 / Loss Curve
    ax2.clear()
    ax2.plot(losses, color='blue', label='损失 Loss')
    ax2.set_title("训练损失变化曲线 / Training Loss Curve", fontsize=12)
    ax2.set_xlabel("训练轮次 / Epoch")
    ax2.set_ylabel("损失值 / Loss")
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, max(losses[0], 0.8))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    return ax1, ax2

# -------------------------------------------------
# 7️⃣ 创建动画 / Create Animation
# -------------------------------------------------
ani = animation.FuncAnimation(fig, update, frames=100, interval=120, blit=False, repeat=False)

plt.tight_layout()
plt.show()
# -------------------------------------------------
# ✅ 可选：保存动画 / Optional: Save Animation
# -------------------------------------------------
print("saving...")
ani.save("双月神经网络训练动画_two_moon_training.gif", writer="pillow", fps=15)
print("双月神经网络训练动画_two_moon_training.gif saved")

sys.exit(0)


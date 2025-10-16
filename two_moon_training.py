import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.datasets import make_moons

# 1️⃣ Generate the dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# 2️⃣ Define the neural network
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

# 3️⃣ Prepare the meshgrid for boundary plotting
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# 4️⃣ Create the figure
fig, ax = plt.subplots(figsize=(6,6))
sc = ax.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap="coolwarm", edgecolor='k')
ax.set_title("Neural Network Learning Decision Boundary")

# 5️⃣ Animation update function
def update(frame):
    # Perform one training step
    for _ in range(10):  # speed up training by 10 steps per frame
        optimizer.zero_grad()
        output = net(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Predict on the grid
    with torch.no_grad():
        preds = net(grid).reshape(xx.shape)

    ax.clear()
    ax.contourf(xx, yy, preds, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap="coolwarm", edgecolor='k')
    ax.set_title(f"Epoch {frame*10} | Loss = {loss.item():.4f}")
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1.5, 2)
    return ax,

# 6️⃣ Create animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=False, repeat=False)

plt.show()

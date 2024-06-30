import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import glob

# 定义手势类别
GESTURES = ["triangle", "circle", "cross"]

# 定义数据集类
class GestureDataset(Dataset):
    def __init__(self, data_dir="gesture_data"):
        self.files = glob.glob(f"{data_dir}/*.npy")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(file)
        gesture = file.split('/')[-1].split('_')[0]
        label = GESTURES.index(gesture)
        return torch.tensor(data, dtype=torch.float32), label

# 定义三层卷积神经网络模型
class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 128, 128)
        self.fc2 = nn.Linear(128, len(GESTURES))
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将形状 (batch_size, 128, 6) 转换为 (batch_size, 6, 128)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练和测试模型的函数
def train_test_split_dataset(dataset, test_size=0.2):
    train_size = int(len(dataset) * (1 - test_size))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

def train_model(data_dir, epochs=10, batch_size=4, learning_rate=0.001):
    dataset = GestureDataset(data_dir)
    train_dataset, test_dataset = train_test_split_dataset(dataset, test_size=0.2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GestureCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    print("训练完成")
    
    # 保存模型
    torch.save(model.state_dict(), "gesture_cnn.pth")
    print("模型已保存为 gesture_cnn.pth")

    # 测试模型
    test_model(test_loader, model)

def test_model(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"测试集上的准确率: {100 * correct / total:.2f}%")

# 开始训练和保存模型
train_model(data_dir="gesture_data")
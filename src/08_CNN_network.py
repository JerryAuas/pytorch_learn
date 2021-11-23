# 作  者:UJS.Lijie
# 时  间:2021/11/23 21:24
"""
卷积神经网络
"""
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 数据集
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 创建模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_siz = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_siz, -1)
        x = self.fc(x)
        return x


model = Model()

# 利用显卡训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 损失函数，优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 冲量值momentum


# 训练过程
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        # 首先优化器清零
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, target)

        # backward
        loss.backward()

        # Updata
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


Accuracy_list = []
Epoch_list = []


# 测试过程
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, lables = data
            images, lables = images.to(device), lables.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += lables.size(0)
            correct += (predicted == lables).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))
    Accuracy_list.append(100 * correct / total)


if __name__ == '__main__':
    print(device)
    for epoch in range(10):
        train(epoch)
        test()
        Epoch_list.append(epoch)

# 绘制相关图像
plt.plot(Accuracy_list, Epoch_list)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

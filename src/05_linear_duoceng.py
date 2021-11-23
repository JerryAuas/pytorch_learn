# 作  者:UJS.Lijie
# 时  间:2021/11/22 22:17
import torch
import numpy as np

# 引入数据集
xy = np.loadtxt('../data/diabetes_data.csv.gz', delimiter=' ', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


# 建立模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(9, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# 损失函数，优化器
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练过程
for epoch in range(100):
    # 前馈
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # 反馈
    optimizer.zero_grad()
    loss.backward()

    # 更新
    optimizer.step()

x_test = torch.Tensor([[5.0]])
y_test = model(x_test)
print('y_pred', y_test.data)
# 作  者:UJS.Lijie
# 时  间:2021/11/23 18:29
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


# 准备数据及
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=' ', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('../data/diabetes_data.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=2)


# 设计模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(9, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# 创建损失函数，优化器
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

w_list = []
b_list = []
# 训练模型
if __name__ == '__main__':
    for epoch in range(100):
        for i, (inputs, lables) in enumerate(train_loader):
            # forward
            y_pred = model(inputs)
            loss = criterion(y_pred, lables)
            print(epoch, i, loss.item())
            # w_list.append(model.linear3.weight.item())
            # b_list.append(model.linear3.bias.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            # Updata
            optimizer.step()
    # plt.plot(w_list, b_list)
    # plt.xlabel('W')
    # plt.ylabel('B')
    # plt.show()
# 作  者:UJS.Lijie
# 时  间:2021/11/22 20:14
"""
线性回归预测分类
"""
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 创建数据集，准备
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# 设计使用的模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

# 准备损失函数， 选择优化器
criterion = torch.nn.BCELoss(size_average=False)
optimizerRprop = torch.optim.Rprop(model.parameters(), lr=0.01)

# 创建绘图元祖
w_list = []
b_list = []
loss_list = []
# 训练模型
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    w_list.append(model.linear.weight.item())
    b_list.append(model.linear.bias.item())
    loss_list.append(loss.item())
    # 优化器回零
    optimizerRprop.zero_grad()
    # loss 回调
    loss.backward()
    # 更新步进
    optimizerRprop.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[5.0]])
y_test = model(x_test)
print('y_pred', y_test.data)

# 绘制图标
plt.plot(w_list, b_list)
plt.xlabel('W')
plt.ylabel('b')
plt.show()
plt.plot(w_list, loss_list)
plt.xlabel('W')
plt.ylabel('loss')
plt.show()
# 作  者:UJS.Lijie
# 时  间:2021/11/22 18:27
import torch
import matplotlib.pyplot as plt

# 创建数据集，准备
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[3.0], [5.0], [7.0]])


# 设计使用的模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

# 准备损失函数， 选择优化器
criterion = torch.nn.MSELoss(size_average=False)
optimizerSGD = torch.optim.SGD(model.parameters(), lr=0.01)
optimizerAdagrad = torch.optim.Adagrad(model.parameters(), lr=0.01)
optimizerAdam = torch.optim.Adam(model.parameters(), lr=0.01)
optimizerAdamax = torch.optim.Adamax(model.parameters(), lr=0.01)
optimizerASGD = torch.optim.ASGD(model.parameters(), lr=0.01)
optimizerLBFGS = torch.optim.LBFGS(model.parameters(), lr=0.01)
optimizerRMSprop = torch.optim.RMSprop(model.parameters(), lr=0.01)
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
    # optimizerSGD.zero_grad()
    # optimizerLBFGS.zero_grad()
    optimizerRprop.zero_grad()
    # optimizerAdamax.zero_grad()
    # optimizerAdam.zero_grad()
    # optimizerAdagrad.zero_grad()
    # optimizerASGD.zero_grad()
    # optimizerRMSprop.zero_grad()
    # loss 回调
    loss.backward()
    # 更新步进
    # optimizerSGD.step()
    # optimizerRMSprop.step()
    # optimizerLBFGS.step()
    # optimizerASGD.step()
    # optimizerAdam.step()
    # optimizerAdagrad.step()
    # optimizerAdamax.step()
    optimizerRprop.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
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
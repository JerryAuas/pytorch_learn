# 作  者:UJS.Lijie
# 时  间:2021/11/24 0:06
"""
使用独热方式
one_hot_lookup
RNNCell
"""

import torch

input_size = 4
hidden_size = 4
batch_size = 1

# 准备数据
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [0, 1, 2, 3, 2]

# 定义独热矩阵，分别用每一行代表四个字母中的其中一个
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)


# 定义模型
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        # 转换隐层
        hidden = self.rnncell(input, hidden)
        return hidden

    def init_hidden(self):
        # 生成初始的全零隐层
        return torch.zeros(self.batch_size, self.hidden_size)


model = Model(input_size, hidden_size, batch_size)

# 构造损失和优化函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 训练模型
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = model.init_hidden()
    print('Predicted string :', end='')
    for input, label in zip(inputs, labels):
        hidden = model(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
        # print(hidden)
    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))
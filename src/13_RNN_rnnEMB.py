# 作  者:UJS.Lijie
# 时  间:2021/11/24 5:14
"""
使用Embedding
RNN
"""
import torch

# 初始化数据
num_class = 3
input_size = 4
hidden_size = 8
num_layers = 1
batch_size = 1
seq_len = 5
embedding_size = 10

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [0, 1, 2, 3, 2]
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)


# 创建模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # x = x.view(len(x), 1, -1)
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        # x = x.view(len(x), 1, -1)
        return x.view(-1, num_class)


model = Model()

# 构造损失和优化函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# 训练模型
for epoch in range(15):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels.squeeze(1))
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted:', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss= %.3f' % (epoch + 1, loss.item()))

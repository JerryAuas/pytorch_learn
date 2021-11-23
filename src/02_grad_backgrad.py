# 作  者:UJS.Lijie
# 时  间:2021/11/22 18:08
import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.Tensor([1.0])
w.requires_grad = True
def forward(x):
    return x*w
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
print("Predict (before training)", 4, forward(4).item())
w_list = []
loss_list = []
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w_list.append(w.data)
        loss_list.append(l)
        w.grad.data.zero_()
    print("progress:", epoch, l.item())
print("Predict (after training)", 8, forward(8).item())
plt.plot(w_list, loss_list)
plt.xlabel("w")
plt.ylabel("Loss")
plt.show()

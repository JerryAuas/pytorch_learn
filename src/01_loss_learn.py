# 作  者:UJS.Lijie
# 时  间:2021/11/19 21:05
import numpy as np
import matplotlib.pyplot as plt



"""
loss xianxing
"""
# x_data = [1.0, 2.0, 3.0]
# y_data = [2.0, 4.0, 6.0]
# def forward(x):
#     return w*x
# def loss(x,y):
#     y_pred = forward(x)
#     return (y_pred-y)*(y_pred-y)
# w_list = []
# loss_list = []
# for w in np.arange(0.0, 4.1, 0.1):
#     print('w:', w)
#     l_sum = 0
#     for x_val, y_val in zip(x_data, y_data):
#         y_pred_val = forward(x_val)
#         loss_val = loss(x_val, y_val)
#         l_sum += loss_val
#         print('\t', x_val, y_val, y_pred_val, loss_val)
#     print('MSE=', l_sum/3)
#     w_list.append(w)
#     loss_list.append(l_sum/3)
# plt.plot(w_list, loss_list)
# plt.xlabel('W')
# plt.ylabel('mes')
# plt.show()

"""
loss 梯度下降
"""
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# w = 1.0
# def forward(x):
#     return (x * w)
#
#
# def cost(xs, ys):
#     cost = 0
#     for x, y in zip(xs, ys):
#         y_pred = forward(x)
#         cost += (y_pred - y) ** 2
#     return (cost / len(xs))
#
#
# def gradient(xs, ys):
#     grad = 0
#     for x, y in zip(xs, ys):
#         grad += 2 * x * (x * w - y)
#     return (grad / len(xs))
#
#
# print('Predict (begin training)', 4, forward(4))
# epoch_list = []
# cost_list = []
# for epoch in range(100):
#     cost_val = float(cost(x_data, y_data))
#     grad_val = float(gradient(x_data, y_data))
#     w -= 0.01 * grad_val
#     cost_list.append(cost_val)
#     epoch_list.append(epoch)
#     print('Epoch:', epoch, 'w=:', w, 'loss=:', cost_val)
# print('Predict (after traning)', 4, forward(4))
# plt.plot(epoch_list,cost_list)
# plt.xlabel('W')
# plt.ylabel('cost')
# plt.show()


"""
随机梯度下降
"""

# w = 1.0
# def forward(x):
#     return x*w
# def loss(x, y):
#     y_pred = forward(x)
#     return (y_pred-y)**2
# def gradient(x,y):
#     return 2*x*(x*w-y)
# print('Predict (before training)', 4, forward(4))
# w_list = []
# loss_list = []
# for epoch in range(100):
#     for x, y in zip(x_data, y_data):
#         grad = gradient(x, y)
#         w = w - 0.01*grad
#         print("\tgrad:",x ,y, grad)
#         l = loss(x, y)
#         w_list.append(w)
#         loss_list.append(l)
#     print("progress:", epoch, "w=", w, "loss=", l)
# print('Predict (after training)', 4, forward(4))
# plt.plot(w_list,loss_list)
# plt.xlabel('x')
# plt.ylabel('loss')
# plt.show()

"""
 
"""
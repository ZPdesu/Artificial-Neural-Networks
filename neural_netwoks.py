# -*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt

_author_ = 'Zhu Peihao'


def rand(a, b, i):
    return (b - a) * random.random(i) +a


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def dsigmoid(x):
    return exp(-x) / power((1.0 + exp(-x)), 2)


def dtanh(x):
    return 1.0 - pow(x, 2)


def plot_curve(x, y):
    plt.plot(x, y)
    plt.show()


class NeuralNetwoks:
    '三层反向传播神经网络'
    # 输入层、隐藏层、输出层的节点数
    def __init__(self, ni, nh, no):
        #
        self.ni = ni +1
        self.nh = nh
        self.no = no

        # 神经网络节点向量
        self.ai = ones(self.ni)
        self.ah = ones(self.nh)
        self.ao = ones(self.no)

        # 权重矩阵
        self.wi = zeros((self.ni, self.nh))
        self.wo = zeros((self.nh, self.no))

        # 随机赋值权重
        for i in range(self.ni):
            self.wi[i] = rand(-0.5, 0.5, self.nh)
        for j in range(self.nh):
            self.wo[j] = rand(-1, 1, self.no)

        # 建立动力因子矩阵
        self.ci = zeros((self.ni, self.nh))
        self.co = zeros((self.nh, self.no))

    # 更新函数
    def update(self, input_array):
        if len(input_array) != self.ni -1:
            raise ValueError('与输入层节点数不符')

        # 激活输入层
        self.ai[0: self.ni -1] = input_array

        # 激活隐藏层
        for h in range(self.nh):
            self.ah[h] = dot(self.ai, self.wi[:, h])
            self.ah[h] = tanh(self.ah[h])

        # 激活输出层
        for j in range(self.no):
            self.ao[j] = dot(self.ah, self.wo[:, j])
            self.ao[j] = sigmoid(self.ao[j])

        return self.ao

    def batch_back(self, targets):
        if len(targets) != self.no:
            raise ValueError('与输出层节点数不符')

        change1 = zeros((self.nh, self.no))
        change2 = zeros((self.ni, self.nh))

        # 输出层误差
        output_deltas = zeros(self.no)
        for j in range(self.no):
            error = targets[j] - self.ao[j]
            output_deltas[j] = dsigmoid(self.ao[j]) * error

        # 隐藏层误差
        hidden_deltas = zeros(self.nh)
        for h in range(self.nh):
            error = 0.0
            for j in range(self.no):
                error = error + output_deltas[j] * self.wo[h, j]
            hidden_deltas[h] = dtanh(self.ah[h]) * error

        for h in range(self.nh):
            for j in range(self.no):
                change1[h, j] = output_deltas[j] * self.ah[h]
        for i in range(self.ni):
            for h in range(self.nh):
                change2[i, h] = hidden_deltas[h] * self.ai[i]

        # 计算误差
        error = 0.0
        for i in range(len(targets)):
            error += 0.5 * power((targets[i] - self.ao[i]), 2)
        return error, change1, change2

    def batch_update(self, change1, change2, N):
                self.wo = self.wo + N * change1
                self.wi = self.wi + N * change2

    # 误差反向传播算法
    def back_propagate(self, targets, N):
        if len(targets) != self.no:
            raise ValueError('与输出层节点数不符')

        # 输出层误差
        output_deltas = zeros(self.no)
        for j in range(self.no):
            error = targets[j] - self.ao[j]
            output_deltas[j] = dsigmoid(self.ao[j]) * error

        # 隐藏层误差
        hidden_deltas = zeros(self.nh)
        for h in range(self.nh):
            error = 0.0
            for j in range(self.no):
                error = error + output_deltas[j] * self.wo[h, j]
            hidden_deltas[h] = dtanh(self.ah[h]) * error

        # 更新输出层权重
        for h in range(self.nh):
            for j in range(self.no):
                change = output_deltas[j] * self.ah[h]
                self.wo[h, j] = self.wo[h, j] + N * change

        # 更新输入层权重
        for i in range(self.ni):
            for h in range(self.nh):
                change = hidden_deltas[h] * self.ai[i]
                self.wi[i, h] = self.wi[i, h] + N * change

        # 计算误差
        error = 0.0
        for i in range(len(targets)):
            error += 0.5 * power((targets[i] - self.ao[i]), 2)
        return error

    def weights(self):
        print('输入层权重：')
        for i in range(self.ni):
            print self.wi[i]
        print ('输出层权重')
        for j in range(self.nh):
            print self.wo[j]

    def train(self, patterns, targets, iteration=100, N=0.1):
        error_list = []
        for i in range(iteration):
            error = 0.0
            for p in range(len(patterns)):
                self.update(patterns[p])
                error = error + self.back_propagate(targets[p], N)
            print error
            error_list.append(error)
        return error_list
    
    def batch_train(self, patterns, targets, iteration=100, N=0.1):
        error_list = []
        for i in range(iteration):
            error = 0.0
            change1 = zeros((self.nh, self.no))
            change2 = zeros((self.ni, self.nh))
            for p in range(len(patterns)):
                self.update(patterns[p])
                p_error, p_change1, p_change2 = self.batch_back(targets[p])
                error += p_error
                change1 += p_change1
                change2 += p_change2
            self.batch_update(change1, change2, N)
            print error
            error_list.append(error)
        return error_list


def main(a):
    patterns = array([[ 1.58, 2.32, -5.8], [ 0.67, 1.58, -4.78], [ 1.04, 1.01, -3.63], [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73],
                      [1.39, 3.16, 2.87], [ 1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [ 0.45, 1.33, -4.38], [-0.76, 0.84, -1.96],
               [0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16], [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39],
               [0.74, 0.96, -1.16], [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14], [0.46, 1.49, 0.68],
               [-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69], [1.86, 3.19, 1.51], [1.68, 1.79, -0.87],
               [3.51, -0.22, -1.39], [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99],[0.66, -0.45, 0.08]])
    targets = array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                     [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                     [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])

    n = NeuralNetwoks(3, 5, 3)
    iteration = 200
    if a == 1:
        error_list = n.train(patterns, targets, iteration, N=0.1)
    else:
        error_list = n.batch_train(patterns, targets, iteration, N=0.1)
    x = range(iteration)
    plot_curve(x, error_list)

if '__main__' == __name__:
    main(2)
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Conv_d311(nn.Module):
    def __init__(self):
        super(Conv_d311, self).__init__()
        kernel = [[-1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d312(nn.Module):
    def __init__(self):
        super(Conv_d312, self).__init__()
        kernel = [[0, -1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d313(nn.Module):
    def __init__(self):
        super(Conv_d313, self).__init__()
        kernel = [[0, 0, -1],
                  [0, 1, 0],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d314(nn.Module):
    def __init__(self):
        super(Conv_d314, self).__init__()
        kernel = [[0, 0, 0],
                  [0, 1, -1],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d315(nn.Module):
    def __init__(self):
        super(Conv_d315, self).__init__()
        kernel = [[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, -1]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d316(nn.Module):
    def __init__(self):
        super(Conv_d316, self).__init__()
        kernel = [[0, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d317(nn.Module):
    def __init__(self):
        super(Conv_d317, self).__init__()
        kernel = [[0, 0, 0],
                  [0, 1, 0],
                  [-1, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d318(nn.Module):
    def __init__(self):
        super(Conv_d318, self).__init__()
        kernel = [[0, 0, 0],
                  [-1, 1, 0],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)

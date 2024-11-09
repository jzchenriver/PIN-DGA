import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Conv_d511(nn.Module):
    def __init__(self):
        super(Conv_d511, self).__init__()
        kernel = [[-1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)


class Conv_d512(nn.Module):
    def __init__(self):
        super(Conv_d512, self).__init__()
        kernel = [[0, 0, -1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)


class Conv_d513(nn.Module):
    def __init__(self):
        super(Conv_d513, self).__init__()
        kernel = [[0, 0, 0, 0, -1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)


class Conv_d514(nn.Module):
    def __init__(self):
        super(Conv_d514, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, -1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)


class Conv_d515(nn.Module):
    def __init__(self):
        super(Conv_d515, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -1]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)


class Conv_d516(nn.Module):
    def __init__(self):
        super(Conv_d516, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, -1, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)


class Conv_d517(nn.Module):
    def __init__(self):
        super(Conv_d517, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [-1, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)


class Conv_d518(nn.Module):
    def __init__(self):
        super(Conv_d518, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [-1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)


class Robinsonc3(nn.Module):
    def __init__(self):
        super(Robinsonc3, self).__init__()
        kernel = [[0, -1/4, 0],
                  [-1/4, 1, -1/4],
                  [0, -1/4, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Robinsonr3(nn.Module):
    def __init__(self):
        super(Robinsonr3, self).__init__()
        kernel = [[-1/8, -1/8, -1/8],
                  [-1/8, 1, -1/8],
                  [-1/8, -1/8, -1/8]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Robinsonc5(nn.Module):
    def __init__(self):
        super(Robinsonc5, self).__init__()
        kernel = [[0, -1/12, -1/12, -1/12, 0],
                  [-1/12, 0, 0, 0, -1/12],
                  [-1/12, 0, 1, 0, -1/12],
                  [-1/12, 0, 0, 0, -1/12],
                  [0, -1/12, -1/12, -1/12, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)


class Robinsonr5(nn.Module):
    def __init__(self):
        super(Robinsonr5, self).__init__()
        kernel = [[-1/16, -1/16, -1/16, -1/16, -1/16],
                  [-1/16, 0, 0, 0, -1/16],
                  [-1/16, 0, 1, 0, -1/16],
                  [-1/16, 0, 0, 0, -1/16],
                  [-1/16, -1/16, -1/16, -1/16, -1/16]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)


class Robinsonc7(nn.Module):
    def __init__(self):
        super(Robinsonc7, self).__init__()
        kernel = [[0, 0, -1/16, -1/16, -1/16, 0, 0],
                  [0, -1/16, 0, 0, 0, -1/16, 0],
                  [-1/16, 0, 0, 0, 0, 0, -1/16],
                  [-1/16, 0, 0, 1, 0, 0, -1/16],
                  [-1/16, 0, 0, 0, 0, 0, -1/16],
                  [0, -1/16, 0, 0, 0, -1/16, 0],
                  [0, 0, -1/16, -1/16, -1/16, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=3)


class Robinsonr7(nn.Module):
    def __init__(self):
        super(Robinsonr7, self).__init__()
        kernel = [[-1/24, -1/24, -1/24, -1/24, -1/24, -1/24, -1/24],
                  [-1/24, 0, 0, 0, 0, 0, -1/24],
                  [-1/24, 0, 0, 0, 0, 0, -1/24],
                  [-1/24, 0, 0, 1, 0, 0, -1/24],
                  [-1/24, 0, 0, 0, 0, 0, -1/24],
                  [-1/24, 0, 0, 0, 0, 0, -1/24],
                  [-1/24, -1/24, -1/24, -1/24, -1/24, -1/24, -1/24]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=3)

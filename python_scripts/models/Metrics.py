import torch
from torch import nn
from torch.nn import functional as F
import math


class SphereProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, m=4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0

        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def dist(self, input):
        return torch.sqrt(torch.sum(torch.square(input), dim=1, keepdim=True) + 1e-12)

    def forward(self, X, label):
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        cosine = self.linear(X) / (self.dist(X) * self.dist(self.linear.weight).reshape(1, -1))
        cosine_m = self.mlambda[self.m](cosine)
        theta = math.acos(cosine)
        k = (self.m * theta / math.pi).floor()
        phi = ((-1.0) ** k) * cosine_m - 2 * k
        s = torch.norm(X, 2, 1)

        one_hot = F.one_hot(label.view(-1).long(), self.linear.out_features)
        output = (one_hot * (phi - cosine) / (1 + self.lamb)) + cosine
        output *= s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'


class AddMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s=30.0, m=0.40) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.s = s
        self.m = m

    def dist(self, input):
        return torch.sqrt(torch.sum(torch.square(input), dim=1, keepdim=True) + 1e-12)

    def forward(self, X, label):
        cosine = self.linear(X) / (self.dist(X) * self.dist(self.linear.weight).reshape(1, -1))
        phi = cosine - self.m

        one_hot = F.one_hot(label.view(-1).long(), self.linear.out_features)
        output = torch.where(one_hot == 1, phi, cosine)
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.linear.in_features) \
               + ', out_features=' + str(self.linear.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s=30.0, m=0.40, easy_margin: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def dist(self, input):
        return torch.sqrt(torch.sum(torch.square(input), dim=1, keepdim=True) + 1e-12)

    def forward(self, X, label):
        cosine = self.linear(X) / (self.dist(X) * self.dist(self.linear.weight).reshape(1, -1))
        sine = torch.sqrt(1 - torch.square(cosine))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(label.view(-1).long(), self.linear.out_features)
        output = torch.where(one_hot == 1, phi, cosine)
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.linear.in_features) \
               + ', out_features=' + str(self.linear.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) \
               + ', easy_margin=' + str(self.easy_margin) + ')'
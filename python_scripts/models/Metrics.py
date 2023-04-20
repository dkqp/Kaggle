import torch
from torch import nn
from torch.nn import functional as F
import math

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

# class ArcMarginProduct(nn.Module):
#     def __init__(self, in_features: int, out_features: int, s=30.0, m=0.40, easy_margin: bool = False) -> None:
#         super().__init__()
#         self.linear = nn.Linear(in_features=in_features, out_features=out_features)
#         self.s = s
#         self.m = m
#         self.easy_margin = easy_margin

#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th
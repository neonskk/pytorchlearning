import torch
from torch import nn


class juanwnang(nn.Module):
    def __init__(self):
        super(juanwnang, self).__init__()
    def forward(self,input):
        output = input + 1
        return output

yang = juanwnang()
x = 1
output = yang(x)
print(output)
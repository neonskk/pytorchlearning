import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1,2,3],dtype=torch.float32)
targets = torch.tensor([1,2,5],dtype=torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

loss =L1Loss()
# loss = L1Loss(reduction='sum')
# reduction='sum'代表着求和，然后不除以个数。
result = loss(inputs,targets)
print(result)

loss_mse = nn.MSELoss()
#平方差，就是相减结果平方后除以个数
result_mse = loss_mse(inputs,targets)
print(result_mse)

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x,y)
print(result_cross)
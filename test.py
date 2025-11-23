import lightning as L

from lightning.pytorch import LightningModule, Trainer, LightningDataModule

import torch
import hydra



net = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10),
)
global_tensor = torch.Tensor([1, 2, 3])

class Model1(torch.nn.Module):
    def __init__(self, net, global_tensor):
        super(Model1, self).__init__()
        self.net = net
        self.global_tensor = global_tensor
        
    def forward(self, x):
        return self.net(x)
    
class Model2(LightningModule):
    def __init__(self, net, global_tensor):
        super(Model2, self).__init__()
        self.net = net
        self.global_tensor = global_tensor
        self.global_tensor.fill_(0)
        
    def forward(self, x):
        return self.net(x)
    
# Test whether Model1 and Model2 share the same network parameters
model1 = Model1(net, global_tensor)
model2 = Model2(net, global_tensor)
for p1, p2 in zip(model1.parameters(), model2.parameters()):
    assert p1 is p2
print("Model1 and Model2 share the same network parameters.")

optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)

# Update Model1
input = torch.randn(16, 28 * 28)
labels = torch.randint(0, 10, (16,))
output1 = model1(input)
loss1 = torch.nn.functional.cross_entropy(output1, labels)
loss1.backward()
optimizer1.step()
optimizer1.zero_grad()

# Not update Model2
# test whether Model2's parameters have changed
print(id(net))
print(id(model2.net))
print(id(model1.net))

# Test whether global_tensor in Model1 and Model2 are the same object
assert model1.global_tensor is model2.global_tensor
print(id(model1.global_tensor))
print(id(model2.global_tensor))
print("Model1 and Model2 share the same global tensor.")
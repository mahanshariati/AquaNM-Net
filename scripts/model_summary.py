import torch
from torchinfo import summary

from aqua_nmnet.models.model import AquaNMNet


model = AquaNMNet(48)
ourArchitecture = summary(model, input_data=(torch.randn(1,3,256,256), [5]*13), depth=4)
print(ourArchitecture)

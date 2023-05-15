"""───────────────┳────────────────────────────────────────────────────────────────────────────────┓
│ Company:        │ Institute of Semiconductors, CAS
│ Engineer:       │ Zhang Ming
│ Create Date:    │ 2023-2-26  14:20
│ Project Name:   │ Quantization
│ File Name:      │ user_model
│ Revision:       │ 1.0
│ Tool Versions:  │ PyCharm
┣─────────────────╋────────────────────────────────────────────────────────────────────────────────┫
│ Description:    │ customized models
┗─────────────────┻──────────────────────────────────────────────────────────────────────────────"""

import torch
import torch.nn as nn
from torchinfo import summary

"""────────────────────────────────────────────────────────────────────────────────────────────────┓
                                            LeNet
┗────────────────────────────────────────────────────────────────────────────────────────────────"""


class LeNet(nn.Module):
    # initial
    def __init__(self, has_bias = True, dataset = "MNIST"):
        super(LeNet, self).__init__()
        # dataset
        self.dataset = dataset
        # dataset
        self.input_channel = None
        # dataset
        self.padding = None
        # dataset
        self.input_size = None
        # bias
        self.has_bias = has_bias
        # define parameter
        if self.dataset == "MNIST":
            self.input_channel = 1
            self.input_size = 28
            self.padding = 2
        elif self.dataset == "CIFAR10":
            self.input_channel = 3
            self.input_size = 32
            self.padding = 0

        # define feature layer
        self.features = nn.Sequential(
                nn.Conv2d(self.input_channel, 6, 5, padding = self.padding, bias = self.has_bias),
                nn.ReLU(True),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5, bias = self.has_bias),
                nn.ReLU(True),
                nn.MaxPool2d(2)
        )
        # define classifier layer
        self.classifier = nn.Sequential(
                nn.Linear(400, 120, bias = self.has_bias),
                nn.ReLU(True),
                nn.Linear(120, 84, bias = self.has_bias),
                nn.ReLU(True),
                nn.Linear(84, 10, bias = self.has_bias)
        )

    # forward
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


"""────────────────────────────────────────────────────────────────────────────────────────────────┓
                                        main procedure
┗────────────────────────────────────────────────────────────────────────────────────────────────"""

if __name__ == "__main__":
    device = "cuda:0"
    model = LeNet(has_bias = True, dataset = "CIFAR10").to(device)
    print(model)
    input_shape = (1, 3, 32, 32)
    summary(model, input_shape)
    x = torch.randn(1, 3, 224, 224).to(device)
    y = model(x)

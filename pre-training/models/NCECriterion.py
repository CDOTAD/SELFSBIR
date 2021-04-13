import torch
import torch.nn as nn


class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        #label = torch.zeros([x.shape[0]]).long().to(x.device)
        return self.criterion(x, label)


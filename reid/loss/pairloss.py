from __future__ import absolute_import

import torch
from torch import nn

from reid.evaluator import accuracy


class PairLoss(nn.Module):
    def __init__(self, sampling_rate=3):
        super(PairLoss, self).__init__()
        self.sampling_rate = sampling_rate
        self.BCE = nn.BCELoss()
        self.BCE.size_average = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, score, tar_probe, tar_gallery):
        cls_Size = score.size()  # torch.Size([4, 4])
        N_probe = cls_Size[0]  # N_probe = 4
        N_gallery = cls_Size[1]  # N_gallery = 4

        tar_gallery = tar_gallery.unsqueeze(1)  # torch.Size([1, 4]) ==> tensor([[36, 29, 71, 16]], device='cuda:0')
        tar_probe = tar_probe.unsqueeze(0)  # torch.Size([4, 1])
        mask = tar_probe.expand(N_probe, N_gallery).eq(tar_gallery.expand(N_probe, N_gallery))
        mask = mask.view(-1).cpu().numpy().tolist()  # <class 'list'>: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

        score = score.contiguous()
        samplers = score.view(-1)  # tensor([0.5001, 0.5000,..., 0.5001, 0.5000, 0.5001],device='cuda:0')
        labels = torch.Tensor(mask).to(self.device)  # tensor([1., 0., 0., 0,..., 0., 0., 0., 1.],device='cuda:0')

        positivelabel = torch.Tensor(mask)
        negativelabel = 1 - positivelabel
        positiveweightsum = torch.sum(positivelabel)  # tensor(4.)
        negativeweightsum = torch.sum(negativelabel)  # tensor(12.)
        neg_relativeweight = positiveweightsum / negativeweightsum * self.sampling_rate  # tensor(1.)
        weights = (positivelabel + negativelabel * neg_relativeweight)
        weights = weights / torch.sum(weights) / 10

        self.BCE.weight = weights.to(self.device)
        loss = self.BCE(samplers, labels)

        samplers_data = samplers.data
        samplers_neg = 1 - samplers_data  # torch.Size([16])
        samplerdata = torch.cat((samplers_neg.unsqueeze(1), samplers_data.unsqueeze(1)), 1)  # torch.Size([16, 2])

        labeldata = torch.LongTensor(mask).to(self.device)  # tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        prec, = accuracy(samplerdata, labeldata)  # tensor(0.2500, device='cuda:0')

        return loss, prec




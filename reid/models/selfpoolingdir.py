from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init


class SelfPoolingDir(nn.Module):
    def __init__(self, input_num, output_num, feat_fc=None):  # 2048,128
        super(SelfPoolingDir, self).__init__()
        self.input_num = input_num
        self.output_num = output_num

        # Linear_Q
        if feat_fc is None:
            self.featQ = nn.Sequential(nn.Linear(self.input_num, self.output_num),
                                       nn.BatchNorm1d(self.output_num))
            for m in self.featQ.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, mode='fan_out')
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                else:
                    print(type(m))
        else:
            self.featQ = feat_fc

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, probe_value, probe_base):  # (bz/2)*sq*128; (bz/2)*sq*2048
        pro_size = probe_value.size()  # torch.Size([4, 8, 128])
        pro_batch = pro_size[0]  # 4
        pro_len = pro_size[1]    # 8

        # generating Querys
        Qs = probe_base.view(pro_batch * pro_len, -1)  # torch.Size([32, 2048])
        Qs = self.featQ(Qs)  # torch.Size([32, 128])
        # Qs = self.featQ_bn(Qs)
        Qs = Qs.view(pro_batch, pro_len, -1)  # torch.Size([4, 8, 128])
        tmp_K = Qs  # torch.Size([4, 8, 128])
        Qmean = torch.mean(Qs, 1, keepdim=True)  # torch.Size([4, 1, 128])
        Hs = Qmean.expand(pro_batch, pro_len, self.output_num)  # torch.Size([4, 8, 128])

        weights = Hs * tmp_K  # torch.Size([4, 8, 128])
        weights = weights.permute(0, 2, 1)  # torch.Size([4, 128, 8])
        weights = weights.contiguous()
        weights = weights.view(-1, pro_len)  # torch.Size([512, 8])
        weights = self.softmax(weights)  # torch.Size([512, 8])
        weights = weights.view(pro_batch, self.output_num, pro_len)  # torch.Size([4, 128, 8])
        weights = weights.permute(0, 2, 1)  # torch.Size([4, 8, 128])
        pool_probe = probe_value * weights  # torch.Size([4, 8, 128])
        pool_probe = pool_probe.sum(1)  # torch.Size([4, 128])
        pool_probe = pool_probe.squeeze(1)  # 32*128
        """
        pool_probe = torch.mean(probe_value,1)
        pool_probe = pool_probe.squeeze(1) # 32*128
        """

        # pool_probe  Batch x featnum
        # Hs  Batch x hidden_num

        return pool_probe, pool_probe

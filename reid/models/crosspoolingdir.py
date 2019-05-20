from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init


class CrossPoolingDir(nn.Module):

    def __init__(self, input_num, output_num, feat_fc=None):
        super(CrossPoolingDir, self).__init__()
        self.input_num = input_num
        self.output_num = output_num

        # Linear_K
        if feat_fc is None:
            self.featK = nn.Sequential(nn.Linear(self.input_num, self.output_num),
                                       nn.BatchNorm1d(self.output_num))
            for m in self.featK.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_uniform_(m.weight, mode='fan_out')
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                else:
                    print(type(m))
        else:
            self.featK = feat_fc

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, gallery_value, gallery_base, querys):

        gal_size = gallery_value.size()  # torch.Size([4, 8, 128])
        gal_batch = gal_size[0]  # 4
        gal_len = gal_size[1]  # 8

        # Linear self-transorfmation
        Q_size = querys.size()  # torch.Size([4, 128])
        pro_batch = Q_size[0]  # 4
        Q_featnum = Q_size[1]  # 128

        K = gallery_base.view(gal_batch * gal_len, -1)  # torch.Size([32, 2048])
        K = self.featK(K)  # torch.Size([32, 128])
        # K = self.featK_bn(K)
        K = K.view(gal_batch, gal_len, -1)  # torch.Size([4, 8, 128])
        #  K: gal_batch x gal_len x H_featnum
        #  query: pro_batch x H_featnum

        Q = querys.unsqueeze(1)  # torch.Size([4, 1, 128])
        Q = Q.unsqueeze(1)  # torch.Size([4, 1, 1, 128])
        K = K.unsqueeze(0)  # torch.Size([1, 4, 8, 128])

        #  Q: pro_batch x 1 x 1 x Q_featnum
        #  K: 1 x gal_batch x gal_len x Q_featnum

        Q = Q.expand(pro_batch, gal_batch, gal_len, Q_featnum)  # torch.Size([4, 4, 8, 128])
        K = K.expand(pro_batch, gal_batch, gal_len, Q_featnum)  # torch.Size([4, 4, 8, 128])

        QK = Q * K  # torch.Size([4, 4, 8, 128])
        QK = QK.permute(0, 1, 3, 2)  # torch.Size([4, 4, 128, 8])

        # pro_batch x gal_batch x Q_featnum x gal_len
        QK = QK.contiguous()
        QK = QK.view(-1, gal_len)  # torch.Size([2048, 8])
        weights = self.softmax(QK)  # torch.Size([2048, 8])
        weights = weights.view(pro_batch,  gal_batch, Q_featnum, gal_len)  # torch.Size([4, 4, 128, 8])

        # gallery : gal_batch x gal_len x Q_featnum
        gallery_value = gallery_value.permute(0, 2, 1)
        # gallery : gal_batch x Q_featnum x gal_len  torch.Size([4, 128, 8])
        gallery_value = gallery_value.contiguous()
        gallery_value = gallery_value.unsqueeze(0)
        # gallery : 1 x gal_batch x Q_featnum x gal_len  torch.Size([1, 4, 128, 8])
        gallery_value = gallery_value.expand(pro_batch, gal_batch, Q_featnum, gal_len)
        # gallery : pro_batch x gal_batch x Q_featnum x gal_len  torch.Size([4, 4, 128, 8])
        pool_gallery = (weights * gallery_value).sum(3)  # torch.Size([4, 4, 128])
        # pool_gallery = pool_gallery.squeeze(3)

        return pool_gallery

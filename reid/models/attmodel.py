from __future__ import absolute_import
from torch import nn
from reid.models import SelfPoolingDir
from reid.models import CrossPoolingDir
import torch.nn.init as init


class AttModuleDir(nn.Module):

    def __init__(self, input_num, output_num, same_fc=True):  # 2048 ,128
        super(AttModuleDir, self).__init__()

        self.input_num = input_num
        self.output_num = output_num

        # attention modules
        if same_fc:
            self.feat_fc = nn.Sequential(nn.Linear(self.input_num, self.output_num), nn.BatchNorm1d(self.output_num))
            for m in self.feat_fc.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                else:
                    print(type(m))

            self.selfpooling_model = SelfPoolingDir(self.input_num, self.output_num, feat_fc=self.feat_fc)
            self.crosspooling_model = CrossPoolingDir(self.input_num, self.output_num, feat_fc=self.feat_fc)
        else:
            self.selfpooling_model = SelfPoolingDir(self.input_num, self.output_num)
            self.crosspooling_model = CrossPoolingDir(self.input_num, self.output_num)

    def forward(self, x, inputs):  # x(bz*sq*128) input(bz*sq*2048)
        xsize = x.size()  # torch.Size([8, 8, 128])
        sample_num = xsize[0]  # 8

        if sample_num % 2 != 0:
            raise RuntimeError("the batch size should be even number!")

        seq_len = x.size()[1]  # 8
        x = x.view(int(sample_num/2), 2, seq_len, -1)  # torch.Size([4, 2, 8, 128])
        inputs = inputs.view(int(sample_num/2), 2, seq_len, -1)  # torch.Size([4, 2, 8, 2048])
        probe_x = x[:, 0, :, :]  # torch.Size([4, 8, 128])
        probe_x = probe_x.contiguous()
        gallery_x = x[:, 1, :, :]  # torch.Size([4, 8, 128])
        gallery_x = gallery_x.contiguous()

        probe_input = inputs[:, 0, :, :]  # torch.Size([4, 8, 2048])
        probe_input = probe_input.contiguous()
        gallery_input = inputs[:, 1, :, :]  # torch.Size([4, 8, 2048])
        gallery_input = gallery_input.contiguous()

        # self-pooling
        pooled_probe, hidden_probe = self.selfpooling_model(probe_x, probe_input)
        pooled_gallery, hidden_gallery = self.selfpooling_model(gallery_x, gallery_input)

        # cross-pooling
        # gallery_x(4*8*128), gallery_input(4*8*2048), pooled_probe(4*128)
        pooled_gallery_2 = self.crosspooling_model(gallery_x, gallery_input, pooled_probe)  # torch.Size([4, 4, 128])
        pooled_probe_2 = self.crosspooling_model(probe_x, probe_input, pooled_gallery)  # torch.Size([4, 4, 128])

        pooled_probe_2 = pooled_probe_2.permute(1, 0, 2)  # torch.Size([4, 4, 128])
        pooled_probe, pooled_gallery = pooled_probe.unsqueeze(1), pooled_gallery.unsqueeze(0)
        #        4*1*128,         4*4*128,       4*4*128,         1*4*128
        return pooled_probe, pooled_gallery_2, pooled_probe_2, pooled_gallery   # (bz/2) * 128,  (bz/2)*(bz/2)*128

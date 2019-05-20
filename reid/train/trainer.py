from __future__ import print_function, absolute_import
import time
import torch

from reid.evaluator import accuracy
from utils.meters import AverageMeter
import torch.nn.functional as F
import sys
from tensorboardX import SummaryWriter


class BaseTrainer(object):

    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, data_loader, optimizer1, optimizer2):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        precisions1 = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)

            loss, prec_oim, prec_score = self._forward(inputs, targets)
            losses.update(loss.item(), targets.size(0))

            precisions.update(prec_oim, targets.size(0))
            precisions1.update(prec_score, targets.size(0))

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            batch_time.update(time.time() - end)
            end = time.time()
            print_freq = 60
            num_step = len(data_loader)  # 1146
            num_iter = num_step * epoch + i
            self.writer.add_scalar('train/loss_step', losses.val, num_iter)
            self.writer.add_scalar('train/loss_avg', losses.avg, num_iter)
            self.writer.add_scalar('train/prec_pairloss', precisions1.avg, num_iter)
            self.writer.add_scalar('train/prec_oimloss', precisions.avg, num_iter)
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'prec_oim {:.2%} ({:.2%})\t'
                      'prec_score {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              losses.val, losses.avg,
                              precisions.val, precisions.avg,
                              precisions1.val, precisions1.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class SEQTrainer(BaseTrainer):

    def __init__(self, cnn_model, att_model, classifier_model, criterion_veri, criterion_oim, mode, rate, logdir):
        super(SEQTrainer, self).__init__(cnn_model, criterion_veri)
        self.att_model = att_model
        self.classifier_model = classifier_model
        self.regular_criterion = criterion_oim
        self.mode = mode
        self.rate = rate
        self.writer = SummaryWriter(log_dir=logdir)

    def _parse_data(self, inputs):
        imgs, flows, pids, _ = inputs
        imgs = imgs.to(self.device)
        flows = flows.to(self.device)
        inputs = [imgs, flows]

        targets = pids.to(self.device)
        return inputs, targets

    def _forward(self, inputs, targets):

        if self.mode == 'cnn':
            out_feat = self.model(inputs[0], inputs[1], self.mode)

            loss, outputs = self.regular_criterion(out_feat, targets)
            prec, = accuracy(outputs.data, targets.data)
            # prec = prec[0]

            return loss, prec, 0, 0

        elif self.mode == 'cnn_rnn':

            feat, feat_raw = self.model(inputs[0], inputs[1], self.mode)
            featsize = feat.size()  # torch.Size([8, 8, 128])
            featbatch = featsize[0]  # 8
            seqlen = featsize[1]  # 8
            # expand the target label ID loss
            featX = feat.view(featbatch * seqlen, -1)  # torch.Size([64, 128])

            targetX = targets.unsqueeze(1)  # tensor([[36],[36],[29],[29],[71],[71],[16],[16]], device='cuda:0')
            targetX = targetX.expand(featbatch, seqlen)  # torch.Size([8, 8])
            targetX = targetX.contiguous()
            targetX = targetX.view(featbatch * seqlen, -1)  # torch.Size([64, 1])
            targetX = targetX.squeeze(1)  # torch.Size([64])
            loss_id, outputs_id = self.regular_criterion(featX, targetX)  # tensor(4.6052, device='cuda:0')

            prec_id, = accuracy(outputs_id.data, targetX.data)  # tensor(0., device='cuda:0')
            # prec_id = prec_id[0]

            # verification label

            featsize = feat.size()  # torch.Size([8, 8, 128])
            sample_num = featsize[0]
            targets = targets.data  # tensor([36, 36, 29, 29, 71, 71, 16, 16], device='cuda:0')
            targets = targets.view(int(sample_num / 2), -1)  # torch.Size([4, 2])
            tar_probe = targets[:, 0]  # tensor([36, 29, 71, 16], device='cuda:0')
            tar_gallery = targets[:, 1]  # tensor([36, 29, 71, 16], device='cuda:0')

            pooled_probe, pooled_gallery_2, pooled_probe_2, pooled_gallery = self.att_model(feat, feat_raw)

            encode_scores = self.classifier_model(pooled_probe, pooled_gallery_2, pooled_probe_2, pooled_gallery)

            encode_size = encode_scores.size()  # torch.Size([4, 4, 2])
            encodemat = encode_scores.view(-1, 2)  # torch.Size([16, 2])
            encodemat = F.softmax(encodemat)  # torch.Size([16, 2])
            encodemat = encodemat.view(encode_size[0], encode_size[1], 2)  # torch.Size([4, 4, 2])
            encodemat = encodemat[:, :, 1]  # torch.Size([4, 4])

            loss_ver, prec_ver = self.criterion(encodemat, tar_probe, tar_gallery)

            loss = loss_id + 100*loss_ver

            return loss, prec_id, prec_ver
        else:
            raise ValueError("Unsupported loss:", self.criterion)

    def train(self, epoch, data_loader, optimizer1, optimizer2, rate):
        self.att_model.train()
        self.classifier_model.train()
        self.rate = rate
        super(SEQTrainer, self).train(epoch, data_loader, optimizer1, optimizer2)

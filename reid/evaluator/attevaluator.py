from __future__ import print_function, absolute_import
import time
import torch
from torch.autograd import Variable
from utils.meters import AverageMeter
from utils import to_numpy
from utils import to_torch
from .eva_functions import cmc, mean_ap
import numpy as np
import torch.nn.functional as F


def evaluate_seq(distmat, query_pids, query_camids, gallery_pids, gallery_camids, cmc_topk=(1, 5, 10, 20)):
    query_ids = np.array(query_pids)
    gallery_ids = np.array(gallery_pids)
    query_cams = np.array(query_camids)
    gallery_cams = np.array(gallery_camids)

    ##
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    top1 = cmc_scores['allshots'][0]
    top5 = cmc_scores['allshots'][4]
    top10 = cmc_scores['allshots'][9]
    top20 = cmc_scores['allshots'][19]

    return mAP, top1, top5, top10, top20


class ATTEvaluator(object):

    def __init__(self, cnn_model, att_model, classifier_model, mode, criterion):
        super(ATTEvaluator, self).__init__()
        self.cnn_model = cnn_model
        self.att_model = att_model
        self.classifier_model = classifier_model
        self.mode = mode
        self.criterion = criterion
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def extract_feature(self, data_loader):
        print_freq = 50
        self.cnn_model.eval()
        self.att_model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        allfeatures = 0
        allfeatures_raw = 0

        for i, (imgs, flows, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            imgs = to_torch(imgs).to(self.device)
            flows = to_torch(flows).to(self.device)

            with torch.no_grad():
                if i == 0:
                    out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)
                    allfeatures = [out_feat]
                    allfeatures_raw = [out_raw]
                    preimgs = imgs
                    preflows = flows
                elif imgs.size(0) < data_loader.batch_size:
                    flaw_batchsize = imgs.size(0)
                    cat_batchsize = data_loader.batch_size - flaw_batchsize
                    imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                    flows = torch.cat((flows, preflows[0:cat_batchsize]), 0)

                    out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)

                    out_feat = out_feat[0:flaw_batchsize]
                    out_raw = out_raw[0:flaw_batchsize]

                    allfeatures.append(out_feat)
                    allfeatures_raw.append(out_raw)
                else:
                    out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)

                    allfeatures.append(out_feat)
                    allfeatures_raw.append(out_raw)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

        allfeatures = torch.cat(allfeatures, 0)
        allfeatures_raw = torch.cat(allfeatures_raw, 0)
        return allfeatures, allfeatures_raw

    def getcrosspool(self, resfeatures, resraw, pooled_gallery, tranum):
        q_start = 0
        pooled_query_2 = []
        with torch.no_grad():
            for qind, qnum in enumerate(tranum):
                query_feat_tmp = resfeatures[q_start:q_start + qnum, :, :]
                query_featraw_tmp = resraw[q_start:q_start + qnum, :, :]
                pooled_query_2_tmp = self.att_model.crosspooling_model(query_feat_tmp, query_featraw_tmp, pooled_gallery)
                pooled_query_2.append(pooled_query_2_tmp)
                q_start += qnum
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            pooled_query_2 = torch.cat(pooled_query_2, 1)
            torch.cuda.empty_cache()
        return pooled_query_2

    def evaluate(self, query_loader, gallery_loader, queryinfo, galleryinfo):

        self.cnn_model.eval()
        self.att_model.eval()
        self.classifier_model.eval()

        querypid = queryinfo.pid  # <class 'list'>: [74, 20, 90, 151, 1, 69, 84, 149, 5, 111, -1,..., 71, 139, 36]
        querycamid = queryinfo.camid   # [0, 0, 0 ,...., 0]
        querytranum = queryinfo.tranum  # <class 'list'>: [35, 21, 24, 26, 28, 38, 30, 32, 1, ..., 20, 31, 25, 29]
        gallerypid = galleryinfo.pid  # # <class 'list'>: [74, 20, 90, 151, 1, 69, 84, 149, 5, 111, -1,..., 71, 139, 36]
        gallerycamid = galleryinfo.camid  # [1, 1, 1 ,...., 1]
        gallerytranum = galleryinfo.tranum  # <class 'list'>: [19, 11, 23, 25, 20,  12,..2, 27, 17]

        query_resfeatures, query_resraw = self.extract_feature(query_loader)  # torch.Size([2787, 8, 128])
        gallery_resfeatures, gallery_resraw = self.extract_feature(gallery_loader)  # torch.Size([2006, 8, 128])

        querylen = len(querypid)  # 100
        gallerylen = len(gallerypid)  # 100

        # online gallery extraction
        single_distmat = np.zeros((querylen, gallerylen))  # <class 'tuple'>: (100, 100)

        q_start = 0
        pooled_query = []
        with torch.no_grad():
            for qind, qnum in enumerate(querytranum):
                query_feat_tmp = query_resfeatures[q_start:q_start+qnum, :, :]  # torch.Size([35, 8, 128])
                query_featraw_tmp = query_resraw[q_start:q_start+qnum, :, :]  # torch.Size([35, 8, 2048])
                pooled_query_tmp, hidden_query_tmp = self.att_model.selfpooling_model(query_feat_tmp, query_featraw_tmp)  # torch.Size([35, 128])
                pooled_query.append(pooled_query_tmp)
                q_start += qnum
            pooled_query = torch.cat(pooled_query, 0)  # torch.Size([2787, 128])

        g_start = 0
        pooled_gallery = []
        with torch.no_grad():
            for gind, gnum in enumerate(gallerytranum):
                gallery_feat_tmp = gallery_resfeatures[g_start:g_start+gnum, :, :]  # torch.Size([19, 8, 128])
                gallery_featraw_tmp = gallery_resraw[g_start:g_start+gnum, :, :]  # torch.Size([19, 8, 2048])
                pooled_gallery_tmp, hidden_gallery_tmp = self.att_model.selfpooling_model(gallery_feat_tmp, gallery_featraw_tmp)
                # torch.Size([19, 128])
                pooled_gallery.append(pooled_gallery_tmp)
                g_start += gnum
            pooled_gallery = torch.cat(pooled_gallery, 0)  # torch.Size([2006, 128])
        # pooled_query, hidden_query = self.att_model.selfpooling_model_1(query_resfeatures, query_resraw)
        # pooled_gallery, hidden_gallery = self.att_model.selfpooling_model_2(gallery_resfeatures, gallery_resraw)

        pooled_query_2 = self.getcrosspool(query_resfeatures, query_resraw, pooled_gallery, querytranum)
        pooled_query_2 = to_numpy(pooled_query_2)
        torch.cuda.empty_cache()
        pooled_gallery_2 = self.getcrosspool(gallery_resfeatures, gallery_resraw, pooled_query, gallerytranum)
        pooled_gallery_2 = to_numpy(pooled_gallery_2)
        torch.cuda.empty_cache()

        pooled_query_2 = to_torch(pooled_query_2).to(self.device)
        pooled_gallery_2 = to_torch(pooled_gallery_2).to(self.device)

        # q_start = 0
        # pooled_query_2 = []
        # with torch.no_grad():
        #     for qind, qnum in enumerate(querytranum):
        #         query_feat_tmp = query_resfeatures[q_start:q_start+qnum, :, :]
        #         query_featraw_tmp = query_resraw[q_start:q_start+qnum, :, :]
        #         pooled_query_2_tmp = self.att_model.crosspooling_model(query_feat_tmp, query_featraw_tmp, pooled_gallery)
        #         pooled_query_2.append(pooled_query_2_tmp)
        #         q_start += qnum
        #         torch.cuda.empty_cache()
        #
        #     torch.cuda.empty_cache()
        #     pooled_query_2 = torch.cat(pooled_query_2, 1)
        #     torch.cuda.empty_cache()


        # g_start = 0
        # pooled_gallery_2 = []
        # with torch.no_grad():
        #     for gind, gnum in enumerate(gallerytranum):
        #         gallery_feat_tmp = gallery_resfeatures[g_start:g_start+gnum, :, :]  # torch.Size([19, 8, 128])
        #         gallery_featraw_tmp = gallery_resraw[g_start:g_start+gnum, :, :]  # torch.Size([19, 8, 2048])
        #         pooled_gallery_2_tmp = self.att_model.crosspooling_model(gallery_feat_tmp, gallery_featraw_tmp, pooled_query)
        #         # torch.Size([2787, 19, 128])
        #         pooled_gallery_2.append(pooled_gallery_2_tmp)
        #         g_start += gnum
        #         torch.cuda.empty_cache()
        #
        #     torch.cuda.empty_cache()
        #     pooled_gallery_2 = torch.cat(pooled_gallery_2, 1)

        pooled_query_2 = pooled_query_2.permute(1, 0, 2)  # torch.Size([2787, 2006, 128])
        pooled_query, pooled_gallery = pooled_query.unsqueeze(1), pooled_gallery.unsqueeze(0)  # torch.Size([2787, 1, 128])  torch.Size([1, 2006, 128])

        with torch.no_grad():
            encode_scores = self.classifier_model(pooled_query, pooled_gallery_2, pooled_query_2, pooled_gallery)

        encode_scores = to_torch(encode_scores).to(self.device)
        encode_size = encode_scores.size()  # torch.Size([2787, 2006, 2])
        encodemat = encode_scores.view(-1, 2)  # torch.Size([5590722, 2])
        encodemat = F.softmax(encodemat)
        encodemat = encodemat.view(encode_size[0], encode_size[1], 2)  # torch.Size([2787, 2006, 2])

        single_distmat_all = encodemat[:, :, 0]  # torch.Size([2787, 2006])
        single_distmat_all = single_distmat_all.data.cpu().numpy()
        q_start, g_start = 0, 0
        for qind, qnum in enumerate(querytranum):
            for gind, gnum in enumerate(gallerytranum):
                distmat_qg = single_distmat_all[q_start:q_start+qnum, g_start:g_start+gnum]
                # percile = np.percentile(distmat_qg, 20)
                percile = np.percentile(distmat_qg, 20)
                if distmat_qg[distmat_qg <= percile] is not None:
                    distmean = np.mean(distmat_qg[distmat_qg <= percile])
                else:
                    distmean = np.mean(distmat_qg)

                single_distmat[qind, gind] = distmean
                g_start = g_start + gnum
            g_start = 0
            q_start = q_start + qnum

        return evaluate_seq(single_distmat, querypid, querycamid, gallerypid, gallerycamid)

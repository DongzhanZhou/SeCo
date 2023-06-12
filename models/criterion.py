import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SyncLoss(object):
    def __init__(self, sync_margin=1.0):
        self.sync_margin = sync_margin
    def l2_distance(self, feat1, feat2):
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        dis = torch.mean(nn.MSELoss(reduction='none')(feat1, feat2), dim=1)
        return dis
    def __call__(self, outputs, neg_param, gt_ratio):
        vision_feat1 = outputs['vision_feat1']
        vision_feat2 = outputs['vision_feat2']

        # positive pair 1, from separated results
        pos_sep_1 = torch.mean(self.l2_distance(outputs['audio_sep_feat1'], vision_feat1), dim=0)
        pos_sep_2 = torch.mean(self.l2_distance(outputs['audio_sep_feat2'], vision_feat2), dim=0)
        # positive pair 2, from gt specs
        pos_loss = pos_sep_1 + pos_sep_2
        if gt_ratio > 0:
            pos_gt_1 = torch.mean(self.l2_distance(outputs['audio_gt_feat1'], vision_feat1), dim=0)
            pos_gt_2 = torch.mean(self.l2_distance(outputs['audio_gt_feat2'], vision_feat2), dim=0)
            pos_loss = pos_loss + gt_ratio * (pos_gt_1 + pos_gt_2)

        # negative pairs: [0, 0.25], easy; [0.25, 0.75], normal; [0.75, 1.0], hard;
        assert neg_param >= 0.0 and neg_param <= 1.0, "neg_param %.2f exceeds range" %(neg_param)
        if neg_param < 0.25: # easy negative
            neg_dis_1 = torch.mean(F.relu(self.sync_margin - self.l2_distance(outputs['audio_sep_feat1'], vision_feat2)), dim=0)
            neg_dis_2 = torch.mean(F.relu(self.sync_margin - self.l2_distance(outputs['audio_sep_feat2'], vision_feat1)), dim=0)
        else:
            neg_dis_1 = torch.mean(F.relu(self.sync_margin - self.l2_distance(outputs['audio_temp_feat1'], vision_feat1)), dim=0)
            neg_dis_2 = torch.mean(F.relu(self.sync_margin - self.l2_distance(outputs['audio_temp_feat2'], vision_feat2)), dim=0)
        neg_loss = neg_dis_1 + neg_dis_2
        return pos_loss, neg_loss

def temp_contrastive_loss(outputs):
    temp_feat1 = F.normalize(outputs['audio_temp_feat1'], p=2, dim=1)
    temp_feat2 = F.normalize(outputs['audio_temp_feat2'], p=2, dim=1)
    audio_sep_feat1 = F.normalize(outputs['audio_sep_feat1'], p=2, dim=1)
    audio_sep_feat2 = F.normalize(outputs['audio_sep_feat2'], p=2, dim=1)
    pos_dis1 = nn.MSELoss()(temp_feat1, audio_sep_feat1)
    pos_dis2 = nn.MSELoss()(temp_feat2, audio_sep_feat2)
    neg_dis = torch.mean(nn.MSELoss(reduction='none')(audio_sep_feat1, audio_sep_feat2), dim=1)
    neg_dis = torch.mean(F.relu(1.0 - neg_dis))
    pos_dis = pos_dis1 + pos_dis2
    return pos_dis, neg_dis

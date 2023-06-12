import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
import os
from .fast_net import FastNet
from .audio_net import Unet
from .sync_net import ResNet10SyncNet

def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid

def warp_spec(inputs):
    B = inputs[0].shape[0]
    T = inputs[0].shape[3]
    grid_warp = torch.from_numpy(
        warpgrid(B, 256, T, warp=True))
    outputs = [F.grid_sample(item, grid_warp) for item in inputs]
    return outputs

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.001)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.0001)

class Synthesizer(nn.Module):
    def __init__(self, vision_dim=256, fc_dim=64):
        super(Synthesizer, self).__init__()
        self.vision_proj = nn.Linear(vision_dim, fc_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound):
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]
        feat_img = self.vision_proj(feat_img).view(B, 1, C)
        z = torch.bmm(torch.sigmoid(feat_img) * self.scale, feat_sound.view(B, C, -1)) \
            .view(B, 1, *sound_size[2:])
        z = z + self.bias
        return z

class SepNet(nn.Module):
    def __init__(self):
        super(SepNet, self).__init__()
        self.vision_net = FastNet()
        ckpt = torch.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), "fast_pretrained.pth"))
        self.vision_net.load_state_dict(ckpt, strict=True)
        self.audio_net = Unet()
        self.audio_net.apply(weights_init)
        self.synthesizer = Synthesizer(vision_dim=256)
        self.synthesizer.apply(weights_init)
        self.sync_net = ResNet10SyncNet()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, audio, vision, neg_param=-1, gt_specs=None, temp_specs=None):
        video1, video2 = vision
        video_feat1 = self.vision_net(video1)
        video_feat2 = self.vision_net(video2)
        audio_feat = self.audio_net(audio)
        pred1 = torch.sigmoid(self.synthesizer(video_feat1, audio_feat))
        pred2 = torch.sigmoid(self.synthesizer(video_feat2, audio_feat))
        outputs = {'pred1': pred1, 'pred2': pred2}
        if neg_param > 0:
            outputs['vision_feat1'] = video_feat1
            outputs['vision_feat2'] = video_feat2
            audio_sep_feat1 = self.sync_net(audio * pred1)
            audio_sep_feat2 = self.sync_net(audio * pred2)
            outputs['audio_sep_feat1'] = audio_sep_feat1
            outputs['audio_sep_feat2'] = audio_sep_feat2
            if gt_specs is not None:
                gt_spec1, gt_spec2 = gt_specs
                audio_gt_feat1 = self.sync_net(gt_spec1)
                audio_gt_feat2 = self.sync_net(gt_spec2)
                outputs['audio_gt_feat1'] = audio_gt_feat1
                outputs['audio_gt_feat2'] = audio_gt_feat2
            if neg_param > 0.25:
                audio_temp1, audio_temp2 = temp_specs
                audio_temp_feat1, audio_temp_feat2 = self.sync_net(audio_temp1), self.sync_net(audio_temp2)
                outputs['audio_temp_feat1'] = audio_temp_feat1
                outputs['audio_temp_feat2'] = audio_temp_feat2
        return outputs

    def forward_single(self, audio, video):
        video_feat = self.vision_net(video)
        audio_feat = self.audio_net(audio)
        pred = torch.sigmoid(self.synthesizer(video_feat, audio_feat))
        audio_sep_feat = self.sync_net(audio * pred)
        return {'video_feat': video_feat, 'pred': pred, 'audio_sep_feat': audio_sep_feat}

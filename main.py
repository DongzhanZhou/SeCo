import os
import sys
import random
import torch
import numpy as np
import pickle
import librosa
import logging
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mir_eval.separation import bss_eval_sources
import tensorboardX as tbx

from models.model import SepNet
from models.criterion import SyncLoss, temp_contrastive_loss
from dataset.dataset import AudioDataset, ValDataset

def istft_reconstruction(mag, phase, hop_length=259):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)

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

class eval_meter(object):
    def __init__(self):
        self.record = [0.,0.,0.]
        self.N = 0
    def reset(self):
        self.record = [0.,0.,0.]
        self.N = 0
    def display(self, epoch):
        print('Epoch {}, SDR {:.4f} SIR {:.4f} SAR {:.4f}'.format(epoch, self.record[0],self.record[1],self.record[2]))
    def write_results(self, file_path, epoch):
        f = open('{}/val_results.txt'.format(file_path), 'a')
        f.write("epoch\t{}\tSDR\t{:.6f}\tSIR\t{:.6f}\tSAR\t{:.6f}\n".format(epoch, self.record[0], self.record[1], self.record[2]))
        f.close()
    def update(self,batch):
        self.record = [i * self.N for i in self.record]
        B = batch['raw1'].size(0)
        I = 0
        for b in range(B):
            audio1 = batch['raw1'][b,:].numpy()
            audio2 = batch['raw2'][b,:].numpy()
            mag = batch['mag'][b,:].squeeze(0).numpy()
            phase = batch['phase'][b,:].squeeze(0).numpy()
            mask1 = batch['mask1'][b,:].squeeze(0).numpy()
            mask2 = batch['mask2'][b,:].squeeze(0).numpy()
            r1 = istft_reconstruction(mag*mask1,phase)
            r2 = istft_reconstruction(mag*mask2,phase)
            lg = len(r1)
            try:
                result = [np.average(i) for i in \
                          bss_eval_sources(np.asarray([audio1[:lg],audio2[:lg]]), \
                          np.asarray([r1,r2]),False)[:3]]
                self.record = [i+j for i,j in zip(self.record,result)]
                I += 1
            except:
                print('skip')
        self.N += I
        self.record = [i / self.N for i in self.record]

def main():
    parser = argparse.ArgumentParser("SeCo")
    parser.add_argument('--foldN', type=int, default=1, help='index of fold (16/5 division)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--val_freq', type=int, default=5, help='evaluation frequency')
    parser.add_argument('--exp_name', type=str, default='', help='experiment name')
    args = parser.parse_args()

    dataset = AudioDataset(args)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,num_workers=args.num_workers)
    val1 = ValDataset(args, state='test')
    valloader1 = DataLoader(val1, batch_size=args.batch_size,num_workers=args.num_workers)

    model = SepNet().cuda()
    sync_criterion = SyncLoss()
    param_groups = [{'params': model.vision_net.parameters(), 'lr': 0.0001}, \
                    {'params': model.audio_net.parameters(), 'lr': 0.001}, \
                    {'params': model.synthesizer.parameters(), 'lr': 0.001}]

    optim = torch.optim.Adam(param_groups)
    model = torch.nn.DataParallel(model)
    save_path = os.path.join('ckpt', 'fold%s' %(args.foldN), args.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = tbx.SummaryWriter(os.path.join('log', 'fold%s' %(args.foldN), args.exp_name))
    iter = 0

    for epoch in range(args.epochs):
        meter = eval_meter()
        model.train()

        for i,data in enumerate(dataloader):
            optim.zero_grad()
            mag, spec1, spec2 = warp_spec([data['mag'], data['spec1'], data['spec2']])

            mask1 = (spec1 > 0.5 * mag).float()
            mask2 = (spec2 > 0.5 * mag).float()

            weight = torch.log1p(mag)
            weight = torch.clamp(weight, 1e-3, 10).cuda()

            log_mag = torch.log(mag + 1e-10).cuda()

            audio_temp1, audio_temp2 = warp_spec([data['normal_temp1'], data['normal_temp2']])
            temp_specs = [torch.log(audio_temp1 + 1e-10).cuda(), torch.log(audio_temp2 + 1e-10).cuda()]
            gt_specs = [torch.log(spec1 + 1e-10).cuda(), torch.log(spec2 + 1e-10).cuda()]

            outputs = model(log_mag, [data['video1'].cuda(), data['video2'].cuda()], 1.0, gt_specs, temp_specs)
            loss1 = F.binary_cross_entropy(outputs['pred1'], mask1.cuda(), weight=weight)
            loss2 = F.binary_cross_entropy(outputs['pred2'], mask2.cuda(), weight=weight)
            sep_loss = (loss1 + loss2) * 0.5

            gt_ratio = max(0.1, 0.9 ** ((epoch * len(dataloader) + i) / 100.0))
            pos_loss, neg_loss = sync_criterion(outputs, neg_param=0.1, gt_ratio=gt_ratio)
            contra_pos_loss, contra_neg_loss = temp_contrastive_loss(outputs)
            loss = sep_loss + 0.01 * (pos_loss + neg_loss) + 0.01 * (contra_pos_loss + contra_neg_loss)

            losses = {'loss_mask': sep_loss.item(), 'pos_inter': pos_loss.item(), \
                'neg_inter': neg_loss.item(), 'pos_temp': contra_pos_loss.item(), \
                'neg_temp': contra_neg_loss.item()}

            loss.backward()
            optim.step()
            writer.add_scalars('Train/Losses',losses,iter)
            iter += 1
            
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))

        model.eval()
        meter = eval_meter()
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                for data in valloader1:
                    log_mag = torch.log(warp_spec([data['mag']])[0] + 1e-10)
                    outputs = model(log_mag, [data['video1'].cuda(), data['video2'].cuda()])
                    out1, out2 = outputs['pred1'], outputs['pred2']
                    grid_unwarp = torch.from_numpy(warpgrid(log_mag.shape[0], 512, log_mag.shape[3], warp=False)).cuda()
                    out1 = F.grid_sample(out1, grid_unwarp)
                    out2 = F.grid_sample(out2, grid_unwarp)
                    data['mask1'] = out1.detach().cpu()
                    data['mask2'] = out2.detach().cpu()
                    meter.update(data)
            meter.display(epoch)
            sdr, sir, sar = meter.record
            writer.add_scalars("Test", {'SDR':sdr,'SIR':sir,'SAR':sar}, epoch)
            meter.write_results(save_path, epoch)
    
    print("Finished training")

if __name__ == '__main__':
    main()
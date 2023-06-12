from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from PIL import Image
import pickle
import random
import numpy as np
import librosa
from .video_transforms import build_transforms

class ValDataset(Dataset):
    def __init__(self, args, state='val'):
        sample_file = pickle.load(open('data/fold{}/{}.pkl'.format(args.foldN, state), 'rb'))
        self.data = sample_file['data']
        self.samples = sample_file['samples']
        self.transforms = build_transforms(is_train=False)
        self.frame_path = args.frame_path
        
        class_pool = {}
        for i,item in enumerate(self.data):
            category = item['category']
            class_pool.setdefault(category, [])
            class_pool[category].append(i)
        self.class_pool = class_pool

    def __len__(self):
        return len(self.samples)

    def cut_audio(self, audio, start):
        length = len(audio)
        audio = audio[start:start + 11025*6]
        return audio, float(start) / length

    def read_images(self, video_path, start):
        video_length = max([int(x[:-4]) for x in os.listdir(video_path)])
        video_start = max(int(round(start * video_length)), 1)
        images = []
        for i in np.arange(0, 72)[::3]:
            img_path = os.path.join(video_path, '%06d.jpg' %(i + video_start))
            img = Image.open(img_path)
            images.append(img)
        return images

    def __getitem__(self, idx):
        i, start1 = self.samples[idx][0]
        j, start2 = self.samples[idx][1]
        audio1, class1, index1 = self.data[i]['audio'], self.data[i]['category'], self.data[i]['index']
        audio2, class2, index2 = self.data[j]['audio'], self.data[j]['category'], self.data[j]['index']
        temp_i = random.sample(self.class_pool[class1], 1)[0]
        temp_j = random.sample(self.class_pool[class2], 1)[0]
        audio_temp1, class_temp1 = self.data[temp_i]['audio'], self.data[temp_i]['category']
        audio_temp2, class_temp2 = self.data[temp_j]['audio'], self.data[temp_j]['category']
        assert class_temp1 == class1 and class_temp2 == class2, "%s-%s, %s-%s" %(class_temp1, class1, class_temp2, class2)

        audio_sample1, video_start1 = self.cut_audio(audio1, start1)
        audio_sample2, video_start2 = self.cut_audio(audio2, start2)
        audio_normal_temp1, _ = self.cut_audio(audio_temp1, len(audio_temp1) // 2)
        audio_normal_temp2, _ = self.cut_audio(audio_temp2, len(audio_temp2) // 2)

        video_path1 = os.path.join(self.frame_path, "{}/{}.mp4".format(class1, index1))
        video_path2 = os.path.join(self.frame_path, "{}/{}.mp4".format(class2, index2))

        video1 = self.transforms(self.read_images(video_path1, video_start1))
        video2 = self.transforms(self.read_images(video_path2, video_start2))

        spec1 = np.abs(librosa.stft(audio_sample1, n_fft=1022, hop_length=259))
        spec2 = np.abs(librosa.stft(audio_sample2, n_fft=1022, hop_length=259))
        spec_mix = librosa.stft(audio_sample1 + audio_sample2, n_fft=1022, hop_length=259)
        mag = np.abs(spec_mix)
        phase = np.angle(spec_mix)

        spec_normal_temp1 = np.abs(librosa.stft(audio_normal_temp1, n_fft=1022, hop_length=259))
        spec_normal_temp2 = np.abs(librosa.stft(audio_normal_temp2, n_fft=1022, hop_length=259))

        return {'raw1':torch.Tensor(audio_sample1),
                'raw2':torch.Tensor(audio_sample2),
                'mag':torch.Tensor(mag).unsqueeze(0),
                'phase': torch.Tensor(phase).unsqueeze(0),
                'spec1':torch.Tensor(spec1).unsqueeze(0),
                'spec2':torch.Tensor(spec2).unsqueeze(0),
                'normal_temp1': torch.Tensor(spec_normal_temp1).unsqueeze(0),
                'normal_temp2': torch.Tensor(spec_normal_temp2).unsqueeze(0),
                'video1':video1,
                'video2':video2}

class AudioDataset(Dataset):
    def __init__(self, args, sample_times=10):
        self.data = pickle.load(open('data/fold{}/train.pkl'.format(args.foldN),'rb'))
        self.sample_times = sample_times
        self.max_length = len(self.data)
        self.transforms = build_transforms(is_train=True)
        class_pool = {}
        for i,item in enumerate(self.data):
            category = item['category']
            class_pool.setdefault(category, [])
            class_pool[category].append(i)
        self.class_pool = class_pool
    def __len__(self):
        return int(len(self.data) * self.sample_times)

    def cut_audio(self, audio):
        length = len(audio)
        if length > 11025*18:
            start = random.randint(11025*6, length-11025*12)
        else:
            start = random.randint(0, length-11025*6)
        audio = audio[start:start + 11025*6]
        return audio, float(start) / length

    def read_images(self, video_path, start):
        video_length = max([int(x[:-4]) for x in os.listdir(video_path)])
        video_start = max(int(round(start * video_length)), 1)
        images = []
        for i in np.arange(0, 72)[::3]:
            img_path = os.path.join(video_path, '%06d.jpg' %(i + video_start))
            img = Image.open(img_path)
            images.append(img)
        return images

    def __getitem__(self, idx):
        flag = True
        while flag:
            i,j = random.randint(0,self.max_length-1),random.randint(0,self.max_length-1)
            if self.data[i]['category'] != self.data[j]['category']: flag = False
        audio1, class1, index1 = self.data[i]['audio'], self.data[i]['category'], self.data[i]['index']
        audio2, class2, index2 = self.data[j]['audio'], self.data[j]['category'], self.data[j]['index']
        temp_i = random.sample(self.class_pool[class1], 1)[0]
        temp_j = random.sample(self.class_pool[class2], 1)[0]
        audio_temp1, class_temp1 = self.data[temp_i]['audio'], self.data[temp_i]['category']
        audio_temp2, class_temp2 = self.data[temp_j]['audio'], self.data[temp_j]['category']
        assert class_temp1 == class1 and class_temp2 == class2, "%s-%s, %s-%s" %(class_temp1, class1, class_temp2, class2)

        audio_sample1, start1 = self.cut_audio(audio1)
        audio_sample2, start2 = self.cut_audio(audio2)
        audio_normal_temp1, _ = self.cut_audio(audio_temp1)
        audio_normal_temp2, _ = self.cut_audio(audio_temp2)
        audio_hard_temp1, _ = self.cut_audio(audio1)
        audio_hard_temp2, _ = self.cut_audio(audio2)

        video_path1 = os.path.join(self.frame_path, "{}/{}.mp4".format(class1, index1))
        video_path2 = os.path.join(self.frame_path, "{}/{}.mp4".format(class2, index2))

        video1 = self.transforms(self.read_images(video_path1, start1))
        video2 = self.transforms(self.read_images(video_path2, start2))

        spec1 = np.abs(librosa.stft(audio_sample1, n_fft=1022, hop_length=259))
        spec2 = np.abs(librosa.stft(audio_sample2, n_fft=1022, hop_length=259))
        spec_mix = librosa.stft(audio_sample1+audio_sample2, n_fft=1022, hop_length=259)
        mag = np.abs(spec_mix)
        phase = np.angle(spec_mix)

        spec_normal_temp1 = np.abs(librosa.stft(audio_normal_temp1, n_fft=1022, hop_length=259))
        spec_normal_temp2 = np.abs(librosa.stft(audio_normal_temp2, n_fft=1022, hop_length=259))
        spec_hard_temp1 = np.abs(librosa.stft(audio_hard_temp1, n_fft=1022, hop_length=259))
        spec_hard_temp2 = np.abs(librosa.stft(audio_hard_temp2, n_fft=1022, hop_length=259))

        return {'raw1':torch.Tensor(audio_sample1),
                'raw2':torch.Tensor(audio_sample2),
                'mag':torch.Tensor(mag).unsqueeze(0),
                'phase': torch.Tensor(phase).unsqueeze(0),
                'spec1':torch.Tensor(spec1).unsqueeze(0),
                'spec2':torch.Tensor(spec2).unsqueeze(0),
                'normal_temp1': torch.Tensor(spec_normal_temp1).unsqueeze(0),
                'normal_temp2': torch.Tensor(spec_normal_temp2).unsqueeze(0),
                'hard_temp1': torch.Tensor(spec_hard_temp1).unsqueeze(0),
                'hard_temp2': torch.Tensor(spec_hard_temp2).unsqueeze(0),
                'video1':video1,
                'video2':video2}

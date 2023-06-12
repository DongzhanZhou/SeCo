# SeCo: Separating Unknown Musical Visual Sounds with Consistency Guidance

Xinchi Zhou, Dongzhan Zhou, Wanli Ouyang, Hang Zhou, Di Hu

## Introduction

In this work, we focus on a more general and challenging scenario in the visual sound separation tasks, namely the separation of unknown musical instruments, where the categories in training and testing phases have no direct overlap with each other. To tackle this new setting, we propose the“Separation-withConsistency” (SeCo) framework, which can accomplish the separation on unknown categories by exploiting the consistency constraints. Experiments demonstrate that our SeCo framework exhibits strong adaptation ability on the novel musical categories and outperforms the baseline methods by a notable margin.

## Environment

```shell
* Python 3.7.4
* Pytorch 1.9.0
* torchvision 0.10.0
```

You should also install opencv-python, pillow, librosa, and mir_eval before you start.

## Dataset Download and Processing

We use the Music-21 dataset to train and evaluate our framework. Please visit [MUSIC-21](https://github.com/roudimit/MUSIC_dataset) to find the YouTube-ID list of videos and how to download them. After the videos are downloaded, we extract frames at 12fps with a resolution of 128x128 and waveforms at 11025Hz from the videos and put them in the following structure:

````
    data
    ├── audio
    |   ├── cello
    │   |   ├── 0000.mp3
    │   |   ├── ...
    │   ├── violin
    │   |   ├── 0000.mp3
    │   |   ├── ...
    │   ├── ...
    |
    └── frames
    |   ├── cello
    │   |   ├── 0000.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── violin
    │   |   ├── 0000.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── ...
    ```
````

We randomly select 16 musical instruments as the training data and use the other 5 categories as the testing data, where each 16/5 division is denoted as 'fold'. Please refer to our paper to find the category distribution of the folds.

To faciliate data loading, we aggregate the audio data into pickle objects and load these pkls into the RAM. For example, training/testing audio of fold1 should be saved in `data/fold1/train.pkl` and `data/fold1/test.pkl`, respectively. The `train.pkl` file is a set of dictionary items, where each item contains keys for 'category' (e.g., 'violin'), 'index' (e.g., '0000'), and 'audio' (audio waveforms of 1d array). The `test.pkl` is a dictionary with two keys, i.e., 'data', and 'samples'. The 'data' value is a list that follows the same structure of the `train.pkl` while the 'samples' value is a list that carries information about testing sample pair. The structure of sample pair is `[['class1/index1', start1], ['class2/index2', start2]]`.

## Experiment

Before the training starts, you should download the pre-trained weights of FastNet from [here](https://drive.google.com/file/d/1vP89-e_WFx9_cr9ldhRgoIIYIINUcQVy/view?usp=drive_link) and put it in the `models` folder.

The training is executed on four GPUs in parallel.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --foldN=1 --exp_name=SeCo
```

## Citation

If you find this repo useful, please consider to cite

```
@inproceedings{zhou2023seco,
  title={SeCo: Separating Unknown Musical Visual Sounds with Consistency Guidance},
  author={Zhou, Xinchi and Zhou, Dongzhan and Ouyang, Wanli and Zhou, Hang and Hu, Di},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5168--5177},
  year={2023}
}
```

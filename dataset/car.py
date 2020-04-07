import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from scipy.io import loadmat


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_mat_frame(path, img_folder):
    results = {}
    tmp_mat = loadmat(path)
    anno = tmp_mat['annotations'][0]
    results['path'] = [os.path.join(img_folder, anno[i][-1][0]) for i in range(anno.shape[0])]
    results['label'] = [anno[i][-2][0, 0] for i in range(anno.shape[0])]
    return results


class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=False, loader=pil_loader, tta=None):

        img_folder = root
        # data = pd.DataFrame.from_dict(get_mat_frame(os.path.join(root, 'devkit', 'cars_.mat')))
        pd_train = pd.DataFrame.from_dict(get_mat_frame(os.path.join(root, 'devkit', 'cars_train_annos.mat'), 'cars_train'))
        pd_test = pd.DataFrame.from_dict(get_mat_frame(os.path.join(root, 'devkit', 'cars_test_annos_withlabels.mat'), 'cars_test'))
        data = pd.concat([pd_train, pd_test])
        data['train_flag'] = pd.Series(data.path.isin(pd_train['path']))
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1

        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.tta = tta

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']

        img = self.loader(os.path.join(self.root, file_path))
        if self.tta is None:
            img = self.transform(img)
        elif self.tta == 'flip':
            img_1 = self.transform(img)
            img_2 = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            img_2 = self.transform(img_2)
            img = torch.stack((img_1, img_2), dim=0)
        else:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_balanced_sampler(self):
        img_labels = np.array(self.imgs['label'].tolist())
        class_sample_count = np.array([len(np.where(img_labels==t)[0]) for t in np.unique(img_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in img_labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        return sampler


if __name__ == '__main__':
    get_data(root='Stanford_Dogs')

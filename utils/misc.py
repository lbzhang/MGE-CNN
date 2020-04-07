import os
import random
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import shutil
import pdb



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# check if dir exist, if not create new folder
def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# ---------------save checkpoint--------------------
def save_checkpoint(state, is_best=False, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# ---------------update meter--------------------
def update_meter(dict_meter, dict_content, batch_size):
    for key, value in dict_meter.items():
        if isinstance(dict_content[key], torch.Tensor):
            value.update(dict_content[key].item(), batch_size)
        else:
            value.update(dict_content[key], batch_size)


# ---------------load checkpoint--------------------
def load_checkpoint(model, pth_file):
    print('==> Reading from model checkpoint..')
    assert os.path.isfile(pth_file), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(pth_file)
    # args.start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']

    pretrained_dict = checkpoint['state_dict']
    model_dict = model.module.state_dict()
    model_dict.update(pretrained_dict)

    # model.module.load_state_dict(checkpoint['state_dict'])
    model.module.load_state_dict(model_dict)
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(pth_file, checkpoint['epoch']))

    # results = {'model': model, 'checkpoint': checkpoint}
    return checkpoint 


# ---------------running mean--------------------
class RunningMean:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1.):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def value(self):
        # if self.count:
        #     return float(self.total_value) / self.count
        # else:
        #     return 0 
        return self.avg

    def __str__(self):
        return str(self.value)

# ---------------more accutate Acc--------------------
class RunningAcc:
    def __init__(self):
        self.avg = 0.
        self.pred = []
        self.tgt = []

    def update(self, logits, tgt):
        pred = torch.argmax(logits, dim=1)
        self.pred.extend(pred.cpu().numpy().tolist())
        self.tgt.extend(tgt.cpu().numpy().tolist())

    @property
    def value(self):
        self.avg = accuracy_score(self.pred, self.tgt)
        return self.avg*100

    def __str__(self):
        return str(self.value)

def set_seeding(seed):
    np.random.seed(seed)
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    torch.cuda.manual_seed_all(seed) # gpu vars
    cudnn.deterministic = True


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

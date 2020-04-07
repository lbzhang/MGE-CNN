import os
import argparse
import pprint
import logging
import yaml
from easydict import EasyDict as edict

import pdb

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# ------------------------------
def parser2dict():
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    # print("Config:\n" + pprint.pformat(cfg))
    return edict(cfg)


# ------------------------------
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        #if k not in b:
        #    raise KeyError('{} is not a valid config key'.format(k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def print_options(opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            # default = self.parser.get_default(k)
            # if v != default:
            #     comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


def cfg_from_file(cfg):
    """Load a config from file filename and merge it into the default options.
    """

    filename=cfg.config
    # args from yaml file
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, cfg)

    # logging.info("Config:\n"+pprint.pformat(cfg))
    # print("Config:\n"+pprint.pformat(cfg))
    # print_options(cfg)
    return cfg


def get_config():
    # args from argparser
    cfg = parser2dict()

    if 'POSE_PARAM_PATH' in os.environ:
        filename = os.environ['POSE_PARAM_PATH'] + '/' + filename
    cfg = cfg_from_file(cfg)
    return cfg


# ----------------------------------------------------------------------------------------
# base
parser.add_argument('-j', '--workers', default=36, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--model', default=None, type=str, metavar='PATH', help='path to model (default: none)')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')
parser.add_argument('--aug', action='store_true', help='whether use data augmentation during training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=0, type=int, help='seeding for all random operation')
parser.add_argument('--config', default='configs/cub_224_baseline.yml', type=str, help='config files')
parser.add_argument('--box_thred', default=0.2, type=float, help='threshod to crop a bounding box')

# train
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--loss_weights', default='[1, 0, 0, 0]', type=str, help='loss weights')
parser.add_argument('--pretrained', default=True, type=bool, help='loss weights')

# test
parser.add_argument('--test-novel-only', action='store_true', help='whether only test on novel classes')
parser.add_argument('--random_sample', type=bool, default=False, help='whether random sample novel class')
parser.add_argument('--trials', type=int, default=1, help='whether random sample novel class')
parser.add_argument('--num_sample', default=1, type=int, metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--tta', default=None, type=str, metavar='N', help='number of novel sample (default: 1)')

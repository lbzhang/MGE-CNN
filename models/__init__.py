from .base import BaseNet
from .local_cam import LocalCamNet
import pretrainedmodels


def get_model(opt):
    if 'baseline' in opt.config:
        net = BaseNet(opt)
    else:
        net = LocalCamNet(opt)
    return net


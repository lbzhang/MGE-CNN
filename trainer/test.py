import os
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import RunningMean, accuracy
import cv2

import utils

import pdb

def show_cam_on_image(img, mask, tgt_file):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(tgt_file, np.uint8(255 * cam))


def update_dest(acc_meter, key, content, target):
    if not key in acc_meter.keys():
        acc_meter[key] = RunningMean()
    acc_tmp = accuracy(content, target, topk=(1,))
    acc_meter[key].update(acc_tmp[0], len(target))


def validate(val_loader, model, cfg=None):
    acc_meter = {}
    p_acc = {}
    model = model.eval()

    with torch.no_grad():
        # pbar = tqdm(val_loader, dynamic_ncols=True, total=len(val_loader))
        # for idx, (input, target) in enumerate(pbar):
        for idx, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            if cfg.tta is None:
                output_dict = model(input)
            else:
                bs, ncrops, c, h, w = input.size()
                output_dict = model(input.view(-1, c, h, w))
                logits_list = output_dict['logits']
                logits_list = [item.view(bs, ncrops, -1).mean(1) for item in logits_list]
                output_dict['logits'] = logits_list

            logits_list = output_dict['logits']

            # ------update acc meter------
            if len(cfg.acc_keys) > 1:
                update_dest(acc_meter, "Accuracy", logits_list[-1], target)
            else:
                acc_keys = cfg.test_acc_keys
                for key, value in zip(acc_keys, logits_list):
                    update_dest(acc_meter, key, value.cuda(), target)

            tmp_str = ''
            for k, v in acc_meter.items(): tmp_str = tmp_str + f"{k}:{v.value:.2f} "
            # pbar.set_description(tmp_str)

        print(f"{idx+1}/{len(val_loader)}--" + tmp_str)



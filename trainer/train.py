import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time

from utils import RunningMean, update_meter, accuracy


def exclude_gt(logit, target, is_log=False):
    logit = F.log_softmax(logit, dim=-1) if is_log else F.softmax(logit, dim=-1)
    mask = torch.ones_like(logit)
    for i in range(logit.size(0)): mask[i, target[i]] = 0

    return mask*logit


def kl_loss(x_pred, x_gt, target):
    # KL Divergence for branch 1 and branch 2
    kl_gt = exclude_gt(x_gt, target, is_log=False)
    kl_pred = exclude_gt(x_pred, target, is_log=True)
    tmp_loss = F.kl_div(kl_pred, kl_gt, reduction='none')
    tmp_loss = torch.exp(-tmp_loss).mean()
    return tmp_loss


# -----------train----------------------------------
def train(train_loader, model, criterion, optimizer, args):
    model = model.train()

    loss_keys = args.loss_keys
    acc_keys = args.acc_keys
    loss_meter = {p: RunningMean() for p in loss_keys}
    acc_meter = {p: RunningMean() for p in acc_keys}

    time_start = time.time()
    for idx, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        # compute output
        output_dict = model(input, target)
        logits = output_dict['logits']

        # -----------------
        loss_values = [criterion['entropy'](logit, target) for k, logit in enumerate(logits)]

        if len(loss_keys) > 1:
            kl_loss_1 = kl_loss(logits[5], logits[2].detach(), target)
            kl_loss_2 = kl_loss(logits[8], logits[5].detach(), target)

            loss_values.extend([kl_loss_1, kl_loss_2])
            loss_values.append(sum(loss_values))
        loss_content = {loss_keys[k]: loss_values[k] for k in range(len(loss_keys))}

        # update acc and loss
        acc_values = [accuracy(logit, target, topk=(1,))[0] for logit in logits] 
        acc_content = {acc_keys[k]: acc_values[k] for k in range(len(acc_keys))}

        update_meter(loss_meter, loss_content, input.size(0))
        update_meter(acc_meter, acc_content, input.size(0))

        tmp_str = ''
        for k, v in loss_meter.items(): tmp_str = tmp_str + f"{k}:{v.value:.4f} "
        tmp_str = tmp_str + "\n"
        for k, v in acc_meter.items(): tmp_str = tmp_str + f"{k}:{v.value:.1f} "

        optimizer.zero_grad()
        loss_values[-1].backward()
        optimizer.step()

    time_eclapse = time.time() - time_start
    print(tmp_str + f"t:{time_eclapse:.1f}s")
    return loss_meter[loss_keys[-1]].value

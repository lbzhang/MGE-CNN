import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import utils
from models import LocalCamNet as Net
from utils import RunningMean, accuracy, RunningAcc
from models import get_model
from dataset import get_loader


def main(opt):
    # model
    utils.ensure_dir(opt.checkpoint)
    model = get_model(opt)
    model = nn.DataParallel(model).cuda()
    utils.load_checkpoint(model, opt.model)

    # dataloader
    val_loader   = get_loader(opt, train=False, shuffle=False)['loader']

    _ = validate(val_loader, model, opt)


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
    # print(f"precise--" + tmp_str_2)

if __name__ == '__main__':
    utils.set_seeding(0)
    cudnn.benchmark = True
    opt = utils.get_config()
    opt.batch_size=1
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    main(opt)

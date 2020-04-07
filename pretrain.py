import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils
# from models import LocalCamNet as Net
# from models import BaseNet as Net
from models import get_model
from trainer import *
from dataset import get_loader

# count_parameters

import pdb

def main(opt):
    # model
    utils.ensure_dir(opt.checkpoint)
    # model = nn.DataParallel(Net(opt.num_classes, opt=opt)).cuda()
    model = get_model(opt)
    model = nn.DataParallel(model).cuda()

    # num_params = utils.count_parameters(model)
    # print(num_params)

    # pdb.set_trace()


    if opt.model: utils.load_checkpoint(model, opt.model)

    # dataloader
    train_loader = get_loader(opt, train=True, shuffle=True)['loader']
    val_loader   = get_loader(opt, train=False, shuffle=False, batch_size=2)['loader']

    # optimizer
    extractor_params = model.module.get_params(prefix='extractor')
    classifier_params = model.module.get_params(prefix='classifier')
    # {'params': extractor_params, 'lr': opt.lr},
    lr_cls = opt.lr
    lr_extractor = 0.1 * lr_cls
    if 'lr_rate' in opt.keys():
        lr_extractor = opt.lr_rate * lr_cls
    params = [
              {'params': classifier_params, 'lr': lr_cls},
              {'params': extractor_params, 'lr': lr_extractor}
              ]
    optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.1, last_epoch=-1)

    # criterion
    entropy_loss = nn.CrossEntropyLoss().cuda()
    criterion = {'entropy': entropy_loss}

    best_loss = 100.
    epoch_start = 0
    if opt.resume:
        checkpoint_dict = utils.load_checkpoint(model, opt.resume)
        epoch_start = checkpoint_dict['epoch']
        print(f'Resuming training process from epoch {epoch_start}...')
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])

    # ------main loop-----
    for epoch in range(epoch_start, opt.epochs):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch: [{epoch+1:d} | {opt.epochs:d}] LR: {lr:f}")

        tmp_loss = train(train_loader, model, criterion, optimizer, opt)
        if (epoch + 1)%10 == 0:
            _ = validate(val_loader, model, opt)

        # ------save checkpoint------
        is_best = tmp_loss < best_loss
        best_loss = min(tmp_loss, best_loss)
        utils.save_checkpoint(
            {'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
            }, is_best, checkpoint=opt.checkpoint)

        if (epoch + 1)%10==0:
            utils.save_checkpoint(
                {'epoch': epoch + 1,
                 'state_dict': model.module.state_dict(),
                 'state_dict': model.module.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                 'scheduler' : scheduler.state_dict()
                }, is_best, checkpoint=opt.checkpoint, filename=f"epoch_{epoch+1}.pth")

    print(f'Best trainning loss: {best_loss}')
    return 0


if __name__ == '__main__':
    cudnn.benchmark = True
    opt = utils.get_config()
    utils.set_seeding(opt.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    main(opt)

import os
import imp
import torch.utils.data
import torchvision.transforms as transforms

import pdb

# imp.load_model()
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

normalize_v2 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform_v1 = transforms.Compose([
                      transforms.RandomResizedCrop(448),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      normalize,
                      ])

train_transform_v2 = transforms.Compose([
                      transforms.Resize(512),
                      transforms.RandomCrop(448),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      normalize,
                      ])

test_transform = transforms.Compose([
                     transforms.Resize(512),
                     transforms.CenterCrop(448),
                     transforms.ToTensor(),
                     normalize
                     ])

test_transform_v2 = transforms.Compose([
                     transforms.Resize((448, 448)),
                     transforms.ToTensor(),
                     normalize
                     ])

test_transform_v3 = transforms.Compose([
                     transforms.Resize(512),
                     transforms.TenCrop(448),
                     lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]),
                     lambda crops: torch.stack([normalize(crop) for crop in crops])
                     ])

test_transform_v4 = transforms.Compose([
                     transforms.Resize(512),
                     transforms.FiveCrop(448),
                     lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]),
                     lambda crops: torch.stack([normalize(crop) for crop in crops])
                     ])

# ----------------------
def get_loader(args, train=True, shuffle=False, batch_size=None):

    data_transform = train_transform_v2 if train else test_transform

    if args.tta in ['ten_crop', 'tencrop']:
        data_transform = test_transform_v3

    if args.tta in ['five_crop', 'fivecrop']:
        data_transform = test_transform_v4

    # data_transform = test_transform_v2

    src_file = ''
    if args.data.lower() in ['stanford-cars', 'car', 'cars']:
        src_file = os.path.join('dataset', 'car.py')
    elif args.data.lower() in ['cub', 'cub-200-2011', 'cub_200_2011']:
        src_file = os.path.join('dataset', 'cub.py')
    elif args.data.lower() in ['aircraft', 'fgvc-aircraft']:
        src_file = os.path.join('dataset', 'aircraft.py')

    ImageLoader = imp.load_source('loader', src_file).ImageLoader

    data = ImageLoader(args.data,
                       transform=data_transform,
                       train=train, tta=args.tta)

    batch_size = args.batch_size
    if not batch_size is None:
        batch_size = batch_size

    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=True)

    return {'data': data, 'loader': data_loader}




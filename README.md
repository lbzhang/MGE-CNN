# MGE-CNN

Pytorch implementation of "ICCV2019-Learning a Mixture of Granularity-Specific Experts for Fine-Grained Categorization"

## Requirement
   - python 3.6
   - tqdm
   - yaml
   - easydict
   - pytorch 1.1
   - pretrainedmodels
   - PIL



## Train
ln -s "Folder of CUB data" CUB-200-2011 \
python pretrain.py --config configs/cub_resnet50.yml 

## Inference
Pretrained model: [link](https://drive.google.com/file/d/1JS8tI0gnBIW-tT97DjL1Rc2kJmorrhM2/view?usp=sharing)

python test.py --config configs/cub_resnet50.yml --model epoch_100.pth

Accuracy: 88.78 %








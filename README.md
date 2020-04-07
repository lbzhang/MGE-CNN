# FGVC-Cam
# branch: mask\_1\_patch\_2

## requirement
- python 3.6.5
- pytorch 1.0 (use pytorch installed on Artemis)
  - module load cuda/9.1.85 
  - module load python/3.6.5
  - module load lapack/3.8.0

## 1-gpu
python pretrain.py --checkpoint results/mask_1_patch_2 --data CUB-200-2011 --loss_weights "[1,1,1,1,1]" --lr 0.001 --batch_size 8 --epochs 100 --gpu_ids 0,1 

python test.py --model results/mask_1_patch_2/checkpoint.pth  --data CUB-200-2011 --batch_size 4 --gpu_ids 0

## 2-gpus
python pretrain.py --checkpoint results/mask_1_patch_2 --data CUB-200-2011 --loss_weights "[1,1,1,1,1]" --lr 0.001 --batch_size 12 --epochs 100 --gpu_ids 0,1 

python test.py --model results/mask_1_patch_2/checkpoint.pth  --data CUB-200-2011 --batch_size 6 --gpu_ids 0,1






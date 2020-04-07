# visualization
python visulize.py --model results/gate/epoch_90.pth  --data CUB-200-2011 --batch_size 1 --gpu_ids 0

# -------------------------------------
# CUB-200-2011
checkpoint='results/cub/resnet101/gate'
python pretrain.py --checkpoint results/cub/gate --data CUB-200-2011 --lr 0.01 --batch_size 6 --epochs 100 --gpu_ids 0
python pretrain.py --checkpoint ${checkpoint} --data CUB-200-2011 --lr 0.01 --batch_size 6 --epochs 100 --gpu_ids 0
python test.py --model results/cub/gate/checkpoint.pth  --data CUB-200-2011 --batch_size 2 --gpu_ids 3


# Stanford-Cars ------------
python pretrain.py --checkpoint results/car/gate --data Stanford-Cars --lr 0.01 --batch_size 6 --epochs 20 --gpu_ids 0
python test.py --model results/car/gate/epoch_80.pth --data Stanford-Cars --batch_size 12 --gpu_ids 0 --tta tencrop


# FGVC-Aircraft ------------
python pretrain.py --model results/aircraft/box_box_box_kl/checkpoint_1.pth --checkpoint results/aircraft/3_expert_part_kl --data FGVC-Aircraft --lr 0.01 --batch_size 6 --epochs 20 --gpu_ids 0
python pretrain.py --checkpoint results/aircraft/3_expert_part_kl_scratch --data FGVC-Aircraft --lr 0.01 --batch_size 6 --epochs 100 --gpu_ids 0
python test.py --model results/aircraft/box_box_box_kl/checkpoint.pth  --data FGVC-Aircraft --batch_size 6 --gpu_ids 0


# Stanford-Dogs ------------
python pretrain.py --resume results/dog/box_box_box_kl/checkpoint.pth --checkpoint results/dog/box_box_box_kl --data Stanford-Dogs --num_classes 120 --loss_weights "[1,1,1,1,1]" --lr 0.001 --batch_size 6 --epochs 100 --gpu_ids 0
python test.py --model results/dog/box_box_box_kl/checkpoint.pth --data Stanford-Dogs --num_classes 120 --batch_size 6 --gpu_ids 0

python pretrain.py --model results/dog/box_box_box_kl/checkpoint_1.pth --checkpoint results/car/box_part_kl --data Stanford-Dogs --loss_weights "[0,1,1,0,1,1,0,1,1,1,0]" --lr 0.001 --batch_size 6 --epochs 50 --gpu_ids 0

# python test.py --model results/mask_patch_box/epoch_100.pth --data CUB-200-2011 --batch_size 4 --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_90.pth --data CUB-200-2011 --batch_size 4 --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_80.pth --data CUB-200-2011 --batch_size 4 --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_70.pth --data CUB-200-2011 --batch_size 4 --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_60.pth --data CUB-200-2011 --batch_size 4 --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_50.pth --data CUB-200-2011 --batch_size 4 --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_40.pth --data CUB-200-2011 --batch_size 4 --gpu_ids 0

# python test.py --model results/mask_patch_box/epoch_100.pth --data CUB-200-2011 --batch_size 2 --tta flip --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_90.pth --data CUB-200-2011 --batch_size 2 --tta flip --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_80.pth --data CUB-200-2011 --batch_size 2 --tta flip --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_70.pth --data CUB-200-2011 --batch_size 2 --tta flip --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_60.pth --data CUB-200-2011 --batch_size 2 --tta flip --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_50.pth --data CUB-200-2011 --batch_size 2 --tta flip --gpu_ids 0
# python test.py --model results/mask_patch_box/epoch_40.pth --data CUB-200-2011 --batch_size 2 --tta flip --gpu_ids 0

python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_100.pth --tta flip
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_100.pth --tta five_crop
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_100.pth --tta ten_crop

python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_90.pth --tta flip
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_90.pth --tta five_crop
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_90.pth --tta ten_crop

python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_80.pth --tta flip
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_80.pth --tta five_crop
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_80.pth --tta ten_crop

python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_70.pth --tta flip
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_70.pth --tta five_crop
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_70.pth --tta ten_crop

python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_60.pth --tta flip
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_60.pth --tta five_crop
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_60.pth --tta ten_crop

python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_50.pth --tta flip
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_50.pth --tta five_crop
python test.py --config configs/aircraft_resnet50.yml --mode results/aircraft/resnet50/epoch_50.pth --tta ten_crop

python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_100.pth --tta flip
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_100.pth --tta five_crop
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_100.pth --tta ten_crop

python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_90.pth --tta flip
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_90.pth --tta five_crop
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_90.pth --tta ten_crop

python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_80.pth --tta flip
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_80.pth --tta five_crop
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_80.pth --tta ten_crop

python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_70.pth --tta flip
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_70.pth --tta five_crop
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_70.pth --tta ten_crop

python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_60.pth --tta flip
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_60.pth --tta five_crop
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_60.pth --tta ten_crop

python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_50.pth --tta flip
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_50.pth --tta five_crop
python test.py --config configs/aircraft_resnet101.yml --mode results/aircraft/resnet101/epoch_50.pth --tta ten_crop


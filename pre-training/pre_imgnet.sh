#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
data_dir='/root/ImageNet/ImageNet/ILSVRC2012'
edge_dir='/root/ImageNet/ImageNet/image_canny'
output_dir='./output/imgnet_rnd_mix_real'
python -m torch.distributed.launch --master_port 12347 --nproc_per_node=4 \
	train.py \
	--img_root ${data_dir} \
	--edge_root ${edge_dir} \
	--dataset sketchy \
	--base-learning-rate 0.03 \
	--alpha 0.999 \
	--crop 0.2 \
	--nce_k 65536 \
	--nce_t 0.07 \
	--num_workers 4 \
	--batch_size 64 \
	--epochs 200 \
	--warmup_epoch 5 \
	--output_dir ${output_dir}

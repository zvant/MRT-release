# N_GPUS=1
# BATCH_SIZE=8
# DATA_ROOT=<YOUR/DATA/ROOT>
# OUTPUT_DIR=./outputs/scenes100/source_only

# CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun \
# --rdzv_endpoint localhost:26506 \
# --nproc_per_node=${N_GPUS} \
# main.py \
# --backbone resnet50 \
# --num_encoder_layers 6 \
# --num_decoder_layers 6 \
# --num_classes 9 \
# --dropout 0.1 \
# --data_root ${DATA_ROOT} \
# --source_dataset cityscapes \
# --target_dataset bdd100k \
# --batch_size ${BATCH_SIZE} \
# --eval_batch_size ${BATCH_SIZE} \
# --lr 2e-4 \
# --lr_backbone 2e-5 \
# --lr_linear_proj 2e-5 \
# --epoch 80 \
# --epoch_lr_drop 60 \
# --mode single_domain \
# --output_dir ${OUTPUT_DIR}


# train source

python main.py --backbone resnet50 --num_encoder_layers 6 --num_decoder_layers 6 --num_classes 3 --dropout 0.1 --data_root /mnt/e/Datasets/MSCOCO2017 --source_dataset mscoco --target_dataset mscoco --batch_size 4 --eval_batch_size 8 --lr 2e-4 --lr_backbone 2e-5 --lr_linear_proj 2e-5 --epoch 3 --epoch_lr_drop 2 --mode single_domain --output_dir outputs/mscoco/source_only --resume r50_deformable_detr_convert.pth --num_workers 6

python main.py --backbone resnet50 --num_encoder_layers 6 --num_decoder_layers 6 --num_classes 3 --dropout 0.1 --data_root /mnt/e/Datasets/MSCOCO2017 --source_dataset mscoco --target_dataset mscoco --batch_size 4 --eval_batch_size 8 --lr 2e-4 --lr_backbone 2e-5 --lr_linear_proj 2e-5 --epoch 3 --epoch_lr_drop 2 --mode eval --output_dir outputs/mscoco/source_only --resume r50_deformable_detr_convert.pth --num_workers 6

python main.py --backbone resnet50 --num_encoder_layers 6 --num_decoder_layers 6 --num_classes 3 --data_root /mnt/e/Datasets/MSCOCO2017 --source_dataset mscoco --target_dataset mscoco --eval_batch_size 8 --mode eval --output_dir /tmp --resume r50_model_best.pth

python eval.py --num_classes 3 --resume r50_model_best.pth --dataset mscoco

CUDA_VISIBLE_DEVICES=4 nohup python main.py --backbone resnet50 --num_encoder_layers 6 --num_decoder_layers 6 --num_classes 3 --dropout 0.1 --data_root /data/add_disk0/zekun/MSCOCO2017 --source_dataset mscoco --target_dataset mscoco --batch_size 3 --eval_batch_size 6 --lr 1e-4 --lr_backbone 2e-5 --lr_linear_proj 2e-5 --epoch 3 --epoch_lr_drop 2 --mode single_domain --output_dir outputs/mscoco/source_only --resume r50_deformable_detr_convert.pth --num_workers 6 &> log_base_r50.log &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --backbone resnet50 --num_encoder_layers 6 --num_decoder_layers 6 --num_classes 3 --dropout 0.1 --data_root /data/add_disk0/zekun/MSCOCO2017 --source_dataset mscoco --target_dataset mscoco --batch_size 3 --eval_batch_size 6 --lr 1e-4 --lr_backbone 2e-5 --lr_linear_proj 2e-5 --epoch 10 --epoch_lr_drop 7 --mode single_domain --output_dir outputs/mscoco/source_only --resume r50_deformable_detr_convert.pth --num_workers 6 &> log_base_r50_ep10.log &

# cross

python main.py --backbone resnet50 --num_encoder_layers 6 --num_decoder_layers 6 --num_classes 3 --dropout 0.0 --data_root /mnt/e/Datasets/MSCOCO2017 --source_dataset mscoco --target_dataset 001 --batch_size 3 --eval_batch_size 6 --lr 2e-5 --lr_backbone 2e-6 --lr_linear_proj 2e-6 --epoch 3 --epoch_lr_drop 2 --mode cross_domain_mae --output_dir outputs/cross/001 --resume r50_model_best.pth

# teach

python main.py --backbone resnet50 --num_encoder_layers 6 --num_decoder_layers 6 --num_classes 3 --dropout 0.0 --data_root /mnt/e/Datasets/MSCOCO2017 --source_dataset mscoco --target_dataset 001 --batch_size 3 --eval_batch_size 6 --lr 2e-5 --lr_backbone 2e-6 --lr_linear_proj 2e-6 --epoch 3 --epoch_lr_drop 2 --mode teaching --output_dir outputs/teach/001 --resume outputs/cross/001/model_last.pth --epoch_retrain 2 --epoch_mae_decay 5 --alpha_dt 0.9 --max_dt 0.6 --teach_box_loss True
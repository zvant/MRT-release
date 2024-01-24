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

python main.py --backbone resnet50 --num_encoder_layers 6 --num_decoder_layers 6 --num_classes 3 --dropout 0.1 --data_root /mnt/e/Datasets/MSCOCO2017 --source_dataset mscoco --target_dataset mscoco --batch_size 2 --eval_batch_size 6 --lr 1e-4 --lr_backbone 2e-5 --lr_linear_proj 2e-5 --epoch 3 --epoch_lr_drop 2 --mode single_domain --output_dir outputs/mscoco/source_only --resume r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth --num_workers 6

# For Faster-RCNN with R-18 backbone, the backbone ResNet-18 is initialized with ImageNet [3] pretrained weights. It is trained on MSCOCO training set for 270,000 iterations with batch size 4 and learning rate 0.0005. It is then finetuned on MSCOCO training set with remapped two object categories for 15,000 iterations with batch size 4 and learning rate 0.0003. 69214 images
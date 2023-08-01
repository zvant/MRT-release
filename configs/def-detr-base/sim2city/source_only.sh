N_GPUS=2
BATCH_SIZE=8
DATA_ROOT=<YOUR/DATA/ROOT>
OUTPUT_DIR=./outputs/def-detr-base/sim2city/source_only

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26503 \
--nproc_per_node=${N_GPUS} \
main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 2 \
--dropout 0.1 \
--data_root ${DATA_ROOT} \
--source_dataset sim10k \
--target_dataset cityscapes \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--epoch 50 \
--epoch_lr_drop 40 \
--mode single_domain \
--output_dir ${OUTPUT_DIR}

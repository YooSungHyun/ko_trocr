NUM_GPU=4
GPU_IDS="0,1,2,3"
PORT_ID=7762
export OMP_NUM_THREADS=8
export WANDB_DISABLED=false
export CUDA_LAUNCH_BLOCKING=1
vision_model_name_or_path="microsoft/trocr-large-printed"
text_model_name_or_path="snunlp/KR-BERT-char16424"
for i in 0
do 
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train_trocr.py \
    --output_dir "output/fold${i}" \
    --seed 42 \
    --train_csv_path "data/preprocess/train.csv" \
    --valid_csv_path "data/preprocess/train.csv" \
    --num_train_epochs 15 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --encoder_model_name_or_path ${vision_model_name_or_path} \
    --decoder_model_name_or_path ${text_model_name_or_path} \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --logging_strategy "steps" \
    --logging_steps "100" \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --dataloader_num_workers "16" \
    --wandb_project "dacon_kyowon" \
    --wandb_name "trocr full data" \
    --wandb_entity "tadev" \
    --label_names "labels" \
    --ddp_find_unused_parameters "True" \
    --fp16
done 
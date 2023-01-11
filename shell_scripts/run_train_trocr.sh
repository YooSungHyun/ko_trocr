NUM_GPU=2
GPU_IDS="0,1"
PORT_ID=5522
export OMP_NUM_THREADS=8
export WANDB_DISABLED=false
export CUDA_LAUNCH_BLOCKING=1
vision_model_name_or_path=""
text_model_name_or_path=""
for i in 0
do 
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID ../train_trocr.py \
    --output_dir "output/fold${i}" \
    --seed 42 \
    --train_csv_path "../../kyowon_ocr_data/preprocess/fold${i}_train.csv" \
    --valid_csv_path "../../kyowon_ocr_data/preprocess/fold${i}_valid.csv" \
    --num_train_epochs 0 \
    --per_device_train_batch_size 0 \
    --per_device_eval_batch_size 0 \
    --gradient_accumulation_steps 0 \
    --encoder_model_name_or_path ${vision_model_name_or_path} \
    --decoder_model_name_or_path ${text_model_name_or_path} \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --logging_strategy "steps" \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --learning_rate 0 \
    --dataloader_num_workers "4" \
    --wandb_project "" \
    --wandb_name "" \
    --wandb_entity "" \
    --label_names "labels" \
    --metric_for_best_model "accuracy" \
    --ddp_find_unused_parameters "True" \
    --predict_with_generate "True" \
    --fp16
done 


# "microsoft/swinv2-tiny-patch4-window8-256" -> 별로...

#"microsoft/swin-base-patch4-window7-224-in22k" -> 베스트

# "microsoft/swinv2-large-patch4-window12-192-22k"
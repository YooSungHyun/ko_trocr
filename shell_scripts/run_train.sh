NUM_GPU=4
GPU_IDS="0,1,2,3"
export OMP_NUM_THREADS=8
vision_model_name_or_path="microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"
text_model_name_or_path="snunlp/KR-BERT-char16424"
for i in 1 2 3 4
do 
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU train.py \
    --output_dir "output/fold${i}" \
    --seed 42 \
    --train_csv_path "data/preprocess/fold${i}_train.csv" \
    --valid_csv_path "data/preprocess/fold${i}_valid.csv" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1\
    --encoder_model_name_or_path ${vision_model_name_or_path} \
    --decoder_model_name_or_path ${text_model_name_or_path} \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --logging_strategy "steps" \
    --load_best_model_at_end \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --dataloader_num_workers "8" \
    --wandb_project "dacon_kyowon" \
    --wandb_name "${i}-sv2bc-Augmentator(total_aug-w12to16-192to256)" \
    --wandb_entity "tadev" \
    --label_names "labels" \
    --metric_for_best_model "accuracy" \
    --ddp_find_unused_parameters "True" \
    --predict_with_generate "True" \
    --fp16
done 


# "microsoft/swinv2-tiny-patch4-window8-256" -> 별로...

#"microsoft/swin-base-patch4-window7-224-in22k" 

# "microsoft/swinv2-large-patch4-window12-192-22k"-> 베스트
# "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft" -> 해볼것

NUM_GPU=4
GPU_IDS="0,1,2,3"
export OMP_NUM_THREADS=8
vision_model_name_or_path="microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"
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
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4\
    --encoder_model_name_or_path ${vision_model_name_or_path} \
    --decoder_model_name_or_path ${text_model_name_or_path} \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --logging_strategy "steps" \
    --logging_steps "100" \
    --load_best_model_at_end \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --dataloader_num_workers "8" \
    --wandb_project "dacon_kyowon" \
    --wandb_name "${i}-sv2bc-Augmentator(0.8-0.5 window12to24-192to384)" \
    --wandb_entity "tadev" \
    --label_names "labels" \
    --metric_for_best_model "accuracy" \
    --ddp_find_unused_parameters "True" \
    --predict_with_generate "True" \
    --generation_num_beams "10" \
    --generation_max_length "32" \
    --fp16
done 

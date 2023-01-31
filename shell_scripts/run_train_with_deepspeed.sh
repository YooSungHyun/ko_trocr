NUM_GPU=4
GPU_IDS="0,1,2,3"
cores=`nproc`
num_workers=4
export OMP_NUM_THREADS=$(($cores-$num_workers))
export WANDB_DISABLED=false
vision_model_name_or_path="microsoft/trocr-large-handwritten"
text_model_name_or_path="snunlp/KR-BERT-char16424"
SEED=42 
for i in 0
do 
CUDA_VISIBLE_DEVICES=$GPU_IDS \
deepspeed train.py \
    --output_dir "output_${SEED}/fold${i}" \
    --seed 42 \
    --train_csv_path "data/preprocess/fold${i}_train.csv" \
    --valid_csv_path "data/preprocess/fold${i}_valid.csv" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --encoder_model_name_or_path ${vision_model_name_or_path} \
    --decoder_model_name_or_path ${text_model_name_or_path} \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --logging_strategy "steps" \
    --logging_steps "100" \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --dataloader_num_workers "4" \
    --label_names "labels" \
    --metric_for_best_model "accuracy" \
    --predict_with_generate "True" \
    --generation_num_beams "1" \
    --generation_max_length "32" \
    --fp16 \
    --wandb_name "trocr-large-handwritten-fold${i}(no_aug)" \
    --wandb_entity "tadev" \
    --wandb_project "dacon_kyowon_final" \
    --deepspeed "ds_config/zero2.json" 
done 
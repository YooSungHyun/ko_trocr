NUM_GPU=4
GPU_IDS="0,1,2,3"
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
vision_model_name_or_path="daekeun-ml/ko-trocr-base-nsmc-news-chatbot"
text_model_name_or_path="daekeun-ml/ko-trocr-base-nsmc-news-chatbot"
SEED=42 
for i in 0
do 
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --master_port=3929 --nproc_per_node $NUM_GPU train_trocr_base.py \
    --output_dir "output_${SEED}/fold${i}" \
    --seed 42 \
    --train_csv_path "data/png_preprocess/fold${i}_train.csv" \
    --valid_csv_path "data/png_preprocess/fold${i}_valid.csv" \
    --num_train_epochs 1 \
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
    --predict_with_generate "True" \
    --generation_num_beams "1" \
    --generation_max_length "32" \
    --wandb_project="dacon_kyowon_final" \
    --wandb_entity="tadev" \
    --wandb_name="(rename_plz)bart_default" \
    --deepspeed "ds_config/zero2.json" \
    --fp16
done 
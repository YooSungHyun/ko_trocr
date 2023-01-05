export OMP_NUM_THREADS=8
for gpu in 0 1 2 3 
do 
CUDA_VISIBLE_DEVICES=$gpu \
python inference_kfold2.py \
    --output_dir "kfold_sub" \
    --seed 42 \
    --test_csv_path "data/preprocess/test.csv" \
    --generation_num_beams "10" \
    --model_name_or_path "output/fold${i}" \
    --per_device_eval_batch_size "64" \
    --dataloader_num_workers "4" \
    --predict_with_generate "True" \
    &
done
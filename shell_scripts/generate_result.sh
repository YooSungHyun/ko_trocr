export OMP_NUM_THREADS=8
for i in 0 1 2 3 4
do 
GPU_IDS="0" \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python inference.py \
    --output_dir "kfold_sub" \
    --seed 42 \
    --test_csv_path "data/preprocess/test.csv" \
    --generation_num_beams "10" \
    --model_name_or_path "output_42/fold${i}" \
    --per_device_eval_batch_size "16" \
    --dataloader_num_workers "4" \
    --predict_with_generate "True" 
done

for i in 0 1 2 3 4
do 
GPU_IDS="0" \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python inference.py \
    --output_dir "kfold_sub" \
    --seed 42 \
    --test_csv_path "data/preprocess/test.csv" \
    --generation_num_beams "10" \
    --model_name_or_path "output_24/fold${i}" \
    --per_device_eval_batch_size "16" \
    --dataloader_num_workers "4" \
    --predict_with_generate "True" 
done

python fold_sum.py \
    --output_dir "kfold_sub" \
    --result_csv_path 'result.csv'
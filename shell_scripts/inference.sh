NUM_GPU=3
GPU_IDS="0,1,2"
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU inference.py \
    --output_dir "./" \
    --seed 42 \
    --test_csv_path "data/preprocess/test.csv" \
    --generation_num_beams "100" \
    --model_name_or_path "output/fold0/" \
    --per_device_eval_batch_size "16" \
    --dataloader_num_workers "12" \
    --predict_with_generate "True" \
    --result_csv_path "sub/7_100.csv"
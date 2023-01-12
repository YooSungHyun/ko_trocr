NUM_GPU=4
export OMP_NUM_THREADS=8
for i in 0 1 2 3 4
do 
GPU_IDS="0,1,2,3" \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU inference_kfold.py \
    --output_dir "kfold_sub" \
    --seed 42 \
    --test_csv_path "data/preprocess/test.csv" \
    --generation_num_beams "10" \
    --model_name_or_path "output/fold${i}" \
    --per_device_eval_batch_size "64" \
    --dataloader_num_workers "4" \
    --predict_with_generate "True" 
done
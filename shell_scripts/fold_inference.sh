cores=`nproc`
num_workers=8
NUM_GPU=4
export OMP_NUM_THREADS=$(($cores-($NUM_GPU*$num_workers)))
GPU_IDS="0,1,2,3"
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU fold_inference.py \
    --output_dir "sub" \
    --seed 42 \
    --test_csv_path "data/preprocess/test.csv" \
    --generation_num_beams "1" \
    --model_name_or_path "output_42/fold0/" \
    --per_device_eval_batch_size "16" \
    --dataloader_num_workers $num_workers \
    --predict_with_generate "True" \
    --fp16 \
    --result_csv_path "sub/0.csv"
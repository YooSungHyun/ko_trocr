# unzip data/raw/open.zip -d data/raw/
# cp data/raw/sample_submission.csv data/

python convert_jpg.py 

python preprocess.py \
    --train_csv_path "data/train.csv" \
    --test_csv_path "data/test.csv" \
    --kfold_n_splits "5" \
    --kfold_shuffle "False"
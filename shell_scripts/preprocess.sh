unzip data/open.zip -d data/

python preprocess.py \
    --train_csv_path "data/train.csv" \
    --test_csv_path "data/test.csv" \
    --kfold_n_splits "5" \
    --kfold_shuffle "False"
# unzip data/raw/open.zip -d data/raw/

# python convert_png.py 

python preprocess.py \
    --train_csv_path "data/train.csv" \
    --test_csv_path "data/test.csv" \
    --kfold_n_splits "5" \
    --kfold_shuffle "False" \
    --kfold_label "label"
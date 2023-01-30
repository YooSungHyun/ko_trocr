import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from literal import Folder, RawDataColumns
from utils.dataset_utils import preprocess_img_path


def main(args):
    train_df = pd.read_csv(args.train_csv_path)
    test_df = pd.read_csv(args.test_csv_path)

    train_df[RawDataColumns.img_path] = train_df[RawDataColumns.img_path].apply(preprocess_img_path)
    test_df[RawDataColumns.img_path] = test_df[RawDataColumns.img_path].apply(preprocess_img_path)
    # ./train/TRAIN_00000.png -> ./data/train/TRAIN_00000.png

    train_df[RawDataColumns.length] = train_df[RawDataColumns.label].str.len()
    train_df_short = train_df[train_df[RawDataColumns.length] == 1].reset_index(drop=True)
    train_df_long = train_df[train_df[RawDataColumns.length] > 1].reset_index(drop=True)
    # 글자수 확인
    # 한글자 label을 short에, 이상을 long에 배치

    if not os.path.exists(Folder.data_preprocess):
        os.mkdir(Folder.data_preprocess)

    kfold = StratifiedKFold(n_splits=args.kfold_n_splits, shuffle=args.kfold_shuffle)

    for fold_num, train_test_idx in enumerate(kfold.split(X=train_df_long, y=train_df_long[RawDataColumns.length])):
        train_idx, test_idx = train_test_idx
        fold_train_long_df = train_df_long.iloc[train_idx].reset_index(drop=True)
        fold_valid_long_df = train_df_long.iloc[test_idx].reset_index(drop=True)
        # 길이를 기준으로 label을 나누어 kfold

        train_csv_name = f"fold{fold_num}_train.csv"
        valid_csv_name = f"fold{fold_num}_valid.csv"

        fold_train_df = pd.concat([train_df_short, fold_train_long_df]).reset_index(drop=True)
        fold_valid_df = fold_valid_long_df.reset_index(drop=True)
        # 한글자 데이터는 훈련에 모두 포함
        # valid set은 길이를 기준으로 stratified

        fold_train_df.to_csv(os.path.join(Folder.data_preprocess, train_csv_name))
        fold_valid_df.to_csv(os.path.join(Folder.data_preprocess, valid_csv_name))

    del test_df[RawDataColumns.label]

    train_df.to_csv(os.path.join(Folder.data_preprocess, "train.csv"))
    test_df.to_csv(os.path.join(Folder.data_preprocess, "test.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_path", type=str, default=None)
    parser.add_argument("--test_csv_path", type=str, default=None)
    parser.add_argument("--kfold_n_splits", type=int, default=5)
    parser.add_argument("--kfold_shuffle", type=bool, default=False)
    args = parser.parse_args()
    main(args)

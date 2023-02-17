import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from literal import Folder, RawDataColumns
from utils.dataset_utils import preprocess_img_path, concat_img_label_short, concat_img_label_long, make_result_dict

import math
import random
import cv2
from collections import deque
from itertools import product
import time


def main(args):
    random.seed(42)

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

    start = time.time()
    os.makedirs(Folder.data_aug_preprocess, exist_ok=True)
    train_list_aug = list()
    file_idx = 0
    goal1_prob = 0.2
    goal2_prob = 0.3
    goal3_prob = 0.2
    """ ********** GOAL 1 START
    GOAL 1. (stage는 만들 문자열의 최종 길이), (sub_stage는 stage를 조합하는 경우의 수를 의미함.)
        HardCoding Prob: 0.2 ('을'과'를'이 엄청 많아, 적당히 작게 설정하여, 있는 label에 대한 데이터 패턴 학습 유도)
        1. label에 있는 단어를 이미지로 조합
            1) 조합은 길이 2 이상부터 진행
            2) length 3의 경우, 사람이 인지하기 쉽게 2+1-> 1+2 으로 진행하도록 함 (sub_stage -1 loop)
        2. sub_stage가 1인 경우에는 조합할 수 있는 최대한을 조합해보도록 함
            1) 제약 1. 무조건 연속되어야함 (가방에 -> 가에 처럼 하나가 빠지면 의미가 달라짐)
            2) 제약 2. random하게 1자리를 섞는거는 이후 포문에 있어서 여기서 처리하지 않음.
        3. sub_stage가 2이상인 경우에는, 가능한 2 세트의 경우의 수를 전부조합
            TODO 1. length 4의 경우 2+1+1은 고려되지 않음. (2+1+1는 3세트임.) (1+3은 2세트)
            TODO 2. 기 조합되어있는 곳에서 찾아보지 않음. (조합의 조합을 하지 않음)
            1) length 4의 텍스트를 만든다면 (sub_stage_df),
            2) length 3의 main_word_df와, length 1의 sub_word_df를 구해서
            3) sub_stage_df 와 둘다 매칭되는 경우에만 조합해서 저장함.
    """
    # ****************** length 2 문자열부터, label이 있는 이미지를 조합으로 새로 생성
    for stage in range(2, max(train_df_long["length"]) + 1):
        # ****************** [sub_stage n]: length 8인 경우, 7+1, 6+2, 5+3, ... ,2+6을 비교하고
        # ****************** [sub_stage 1]: 마지막으로 1자리 텍스트만으로 생성해봄
        for sub_stage in range(stage - 1, 0, -1):
            # label을 조사해볼 sub_stage_df
            sub_stage_df = train_df_long[train_df_long["length"] == stage].reset_index(drop=True)
            # ****************** 가장 많은 label의 20% 만큼 조합
            goal1_aug_cnt = math.ceil(max(sub_stage_df.groupby("label").count()["length"]) * goal1_prob)
            # ****************** sub_stage 1: 1자리의 경우 1자리만으로 경우의 수를 조합합니다.
            if sub_stage == 1:
                for label_word in sub_stage_df["label"].unique():
                    found_indices = list()
                    # ****************** 1자리의 경우 word를 각각 조사
                    label_word = deque(label_word)
                    while label_word:
                        char = label_word.popleft()
                        found_idx_flag = train_df_short["label"].isin([char])
                        if found_idx_flag.any():
                            found_indices.append(train_df_short[found_idx_flag].index)
                        else:
                            # '가족의' 의 경우, '가','의'만 있는 경우 의미와 다르게 합쳐지므로, found_indices를 초기화
                            # 긴 문장의 경우 중간의 텍스트만이라도 있으면, 마지막 등장 조합만 유지될 수 있도록 처리함.
                            # ex) '나는배가고프다' -> '는배가' 만 있는경우에 해당 이미지는 생성됨.
                            #     '는배','고프다' -> '고프다'만 유지됨.
                            found_indices = list()
                    # 한자리 조합의 random 생성의 경우, 두번째 aug에서 처리가능하므로 여기서 처리하지 않음.
                    # ****************** 해당 포문까지 완성되면, found_indices에 1자리의 연속된 가능한 단어가 존재.
                    if len(found_indices) > 1:
                        for combination in random.sample(list(product(*found_indices)), goal1_aug_cnt):
                            selected_df = train_df_short.iloc[list(combination)]
                            concated_img, concated_label = concat_img_label_short(selected_df.values)
                            concated_dict = make_result_dict(f"TRAIN_{file_idx:>06d}.png", concated_label)
                            # ****************** 이미지 출력
                            cv2.imwrite(concated_dict["img_path"], concated_img)
                            train_list_aug.append(concated_dict)
                            file_idx += 1
            else:
                # ****************** sub_stage n: main과 sub의 2가지 조합만 고려하여 조합합니다. (7+1, 6+2....)
                for main_word_cnt in range(sub_stage, 0, -1):
                    main_word_df = train_df[train_df["length"] == main_word_cnt].reset_index(drop=True)
                    sub_word_cnt = stage - main_word_cnt
                    sub_word_df = train_df[train_df["length"] == sub_word_cnt].reset_index(drop=True)
                    for label_word in sub_stage_df["label"].unique():
                        found_indices = list()
                        main_found_word = label_word[:main_word_cnt]
                        sub_found_word = label_word[main_word_cnt:]
                        main_found_idx = main_word_df["label"].isin([main_found_word])
                        sub_found_idx = sub_word_df["label"].isin([sub_found_word])
                        if main_found_idx.any() and sub_found_idx.any():
                            # 합칠게 있는경우
                            found_indices.append(main_word_df[main_found_idx].index)
                            found_indices.append(sub_word_df[sub_found_idx].index)
                            # ****************** 실제로 필요한 이미지와 label을 완성하는 작업
                            for combination in random.sample(list(product(*found_indices)), goal1_aug_cnt):
                                concated_img, concated_label = concat_img_label_long(
                                    main_word_df.loc[combination[0], "img_path"],
                                    sub_word_df.loc[combination[1], "img_path"],
                                    main_word_df.loc[combination[0], "label"],
                                    sub_word_df.loc[combination[1], "label"],
                                )
                                concated_dict = make_result_dict(f"TRAIN_{file_idx:>06d}.png", concated_label)
                                # ****************** 이미지 출력
                                cv2.imwrite(concated_dict["img_path"], concated_img)
                                train_list_aug.append(concated_dict)
                                file_idx += 1

    """ ********** GOAL 2 START
    2세트 조합을 랜덤으로 조합함. 각 길이별 전체 data cnt를 고려하여 생성됨
    ********** """
    # ****************** length 3 문자열부터, label이 있는 이미지를 조합으로 새로 생성
    for stage in range(3, max(train_df_long["length"]) + 1):
        for sub_stage in range(stage - 1, 1, -1):
            # ****************** sub_stage n: main과 sub의 2가지 조합만 고려하여 조합합니다. (7+1, 6+2....)
            for main_word_cnt in range(sub_stage, 0, -1):
                goal2_aug_cnt = (
                    train_df["length"].value_counts().apply(lambda x: math.ceil(x * goal2_prob)).sort_index()
                )
                main_word_df = train_df[train_df["length"] == main_word_cnt].reset_index(drop=True)
                sub_word_cnt = stage - main_word_cnt
                sub_word_df = train_df[train_df["length"] == sub_word_cnt].reset_index(drop=True)
                """unique 쓰기위한 처리-1"""
                # unique_main_words = main_word_df["label"].unique()
                # unique_sub_words = sub_word_df["label"].unique()
                # main_rand_indices = [
                #     random.randrange(0, len(unique_main_words) - 1) for _ in range(goal2_aug_cnt.loc[main_word_cnt])
                # ]
                # sub_rand_indices = [
                #     random.randrange(0, len(unique_sub_words) - 1) for _ in range(goal2_aug_cnt.loc[sub_word_cnt])
                # ]
                """*********************"""
                # goal2_aug_cnt.loc[main_word_cnt]: stage 4를 만들기위해 3+1을 하는경우, 3의 길이*0.2 만큼을 추출
                main_rand_indices = random.sample(range(0, len(main_word_df) - 1), goal2_aug_cnt.loc[main_word_cnt])
                # goal2_aug_cnt.loc[sub_word_cnt]: stage 4를 만들기위해 3+1을 하는경우, 1의 길이*0.2 만큼을 추출
                sub_rand_indices = random.sample(range(0, len(sub_word_df) - 1), goal2_aug_cnt.loc[sub_word_cnt])
                # goal2_aug_cnt.loc[stage] // (stage - 1): stage 4의 경우 1+3, 2+2, 3+1, 5의 경우 1+4, 2+3, 3+2, 4+1
                # 식이므로, (stage - 1) 만큼을 균등 샘플링해야 공정함.
                each_sampled_ratio = goal2_aug_cnt.loc[stage] // (stage - 1)
                for combination in random.sample(
                    list(product(main_rand_indices, sub_rand_indices)), each_sampled_ratio
                ):
                    concated_img, concated_label = concat_img_label_long(
                        main_word_df.loc[combination[0], "img_path"],
                        sub_word_df.loc[combination[1], "img_path"],
                        main_word_df.loc[combination[0], "label"],
                        sub_word_df.loc[combination[1], "label"],
                    )
                    concated_dict = make_result_dict(f"TRAIN_{file_idx:>06d}.png", concated_label)
                    cv2.imwrite(concated_dict["img_path"], concated_img)
                    train_list_aug.append(concated_dict)
                    file_idx += 1

    """ ********** GOAL 3 START
    1세트 조합을 랜덤으로 조합함, 각 길이별 전체 data cnt를 고려하여 생성됨

    label이 중복되지 않는 단일 텍스트 sequence를 랜덤하게 만듬
    ****************************** """
    goal3_aug_cnt = train_df["length"].value_counts().apply(lambda x: math.ceil(x * goal3_prob)).sort_index().loc[2:]
    # ****************** label_len에 대한 cnt만큼 포문을 돌려야 len별 cnt만큼 이미지 생성 가능
    for label_len, cnt in goal3_aug_cnt.items():
        for _ in range(cnt):
            rand_indices = deque(random.sample(range(0, len(train_df_short) - 1), label_len))
            # ****************** unique의 length가 겹치면, 무조건 겹치는 문자열이 발생했다는 이야기.
            while label_len != len(train_df_short.iloc[rand_indices]["label"].unique()):
                # 한놈을 무조건 꺼냄
                idx = rand_indices.popleft()
                if train_df_short.iloc[rand_indices]["label"].isin([train_df_short.iloc[idx]["label"]]).any():
                    # 같은게 한놈이라도 있으면, 무조건 새로 넣음
                    rand_indices.append(random.randrange(0, len(train_df_short) - 1))
                else:
                    # 같은게 없는 놈이었으면 유지
                    rand_indices.append(idx)
                # 해당 while문에서 성능 저하 요소가 분명 존재할 것 같으며,
                # popleft -> append를 통해서, 문자열의 순서의 random성을 좀 더 추가해봤습니다.

            # ****************** 실제로 필요한 이미지와 label을 완성하는 작업
            selected_df = train_df_short.iloc[rand_indices]
            concated_img, concated_label = concat_img_label_short(selected_df.values)

            concated_dict = make_result_dict(f"TRAIN_{file_idx:>06d}.png", concated_label)

            # ****************** 이미지 출력
            cv2.imwrite(concated_dict["img_path"], concated_img)
            train_list_aug.append(concated_dict)
            file_idx += 1
    # ****************** 완성된 녀석들을 확인하기 위한 csv를 작성합니다.
    train_df_aug = pd.DataFrame(train_list_aug)
    train_df_aug.to_csv(os.path.join(Folder.data, "train_aug.csv"), index=False)
    end = time.time()
    print(f"{end - start:.5f} sec")

    os.makedirs(Folder.data_preprocess, exist_ok=True)

    kfold = StratifiedKFold(n_splits=args.kfold_n_splits, shuffle=args.kfold_shuffle)

    for fold_num, train_test_idx in enumerate(kfold.split(X=train_df_long, y=train_df_long[args.kfold_label])):
        train_idx, test_idx = train_test_idx
        fold_train_long_df = train_df_long.iloc[train_idx].reset_index(drop=True)
        fold_valid_long_df = train_df_long.iloc[test_idx].reset_index(drop=True)
        # 길이를 기준으로 label을 나누어 kfold

        train_csv_name = f"fold{fold_num}_train.csv"
        valid_csv_name = f"fold{fold_num}_valid.csv"

        fold_train_df = pd.concat([train_df_aug, train_df_short, fold_train_long_df]).reset_index(drop=True)
        # fold_train_df = pd.concat([train_df_short, fold_train_long_df]).reset_index(drop=True)
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
    parser.add_argument("--kfold_label", type=str, default="length")
    args = parser.parse_args()
    main(args)

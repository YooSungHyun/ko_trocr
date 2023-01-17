import os
from ast import literal_eval
from collections import defaultdict

import pandas as pd
from transformers import HfArgumentParser

from arguments import DatasetsArguments, ModelArguments, MyTrainingArguments
from literal import RawDataColumns


def main(model_args: ModelArguments, dataset_args: DatasetsArguments, training_args: MyTrainingArguments):
    folder = training_args.output_dir
    files = os.listdir(folder)

    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(folder, file))
        df[RawDataColumns.label] = df[RawDataColumns.label].apply(literal_eval)
        df[RawDataColumns.seq_probs] = df[RawDataColumns.seq_probs].apply(literal_eval)
        dfs.append(df)

    ensemble_labels = []
    for i in range(len(dfs[0])):
        labels = defaultdict(int)
        for df in dfs:
            for beam_label, beam_prob in zip(df["label"][i], df["seq_probs"][i]):
                # df['label'][i] -> ['낱말', '날말', '남말', '날망', '낭말']
                labels[beam_label] += beam_prob
        sorted_label = sorted(labels.items(), key=lambda x: x[1], reverse=True)
        ensemble_labels.append(sorted_label[0][0])

    sub = pd.read_csv(dataset_args.submission_csv_path)
    sub[RawDataColumns.label] = ensemble_labels
    sub.to_csv(dataset_args.result_csv_path, index=False)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args=model_args, dataset_args=dataset_args, training_args=training_args)

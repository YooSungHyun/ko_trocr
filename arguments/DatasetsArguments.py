from dataclasses import dataclass, field


@dataclass
class DatasetsArguments:
    train_csv_path: str = field(default=None)
    valid_csv_path: str = field(default=None)
    test_csv_path: str = field(default=None)
    result_csv_path: str = field(default=None)
    submission_csv_path: str = field(default="data/sample_submission.csv")

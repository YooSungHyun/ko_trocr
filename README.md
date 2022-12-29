# kyowon_ocr

## environment
```
python3.8
```

## install
```
bash shell_scripts/install_bash.sh
pip install -re requirements.txt
```

## run
- data 폴더 아래에 open.zip (데이콘 데이터) 가 있어야함
```
cd data
unzip open.zip
cd ..
bash shell_scripts/preprocess.sh
bash shell_scripts/run_train.sh
bash shell_scropts/inference.sh
```

## dirs 
```
├── data
│   ├── open.zip
│   ├── preprocess
│   ├── sample_submission.csv
│   ├── test
│   ├── test.csv
│   ├── train
│   └── train.csv
├── arguments
│   ├── DatasetsArguments.py
│   ├── __init__.py
│   ├── ModelArguments.py
│   ├── MyTrainingArguments.py
│   └── __pycache__
├── utils
│   ├── data_collator.py
│   ├── dataset_utils.py
│   ├── __init__.py
│   ├── __pycache__
│   └── training_utils.py
├── shell_scripts
│   ├── inference.sh
│   ├── install_base.sh
│   ├── pid_kill.sh
│   ├── preprocess.sh
│   └── run_train.sh
├── LICENSE
├── literal.py
├── output
├── preprocess.py
├── README.md
├── requirements.txt
├── inference.py
└── train.py
```

## rule
- 브랜치를 파서 main에 pr
- 브랜치는 pr 별로 생성 및 삭제
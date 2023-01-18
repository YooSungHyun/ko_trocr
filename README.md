# kyowon_ocr

## environment
```
python3.8
cuda=11.7
Ubuntu 18.04.6 LTS
```
## dirs 
```
├── arguments
│   ├── __init__.py
│   ├── DatasetsArguments.py
│   ├── ModelArguments.py
│   └── MyTrainingArguments.py
├── data
│   ├── open.zip
│   └── README.md
├── inference.py
├── shell_scripts
│   ├── generate_result.sh # 재현용 inference script
│   ├── inference_kfold.sh
│   ├── preprocess.sh
│   ├── run_train_ensemble.sh # 재현용 train script
│   └── run_train.sh
├── utils
│   ├── __init__.py
│   ├── augmentation.py
│   ├── data_collators.py
│   ├── dataset_utils.py
│   └── training_utils.py
├── LICENSE
├── literal.py
├── preprocess.py
├── README.md
└── train.py
```
## setting environments
### 1. make & activate venv
```
$ python3.8 -m venv .venv
$ source .venv/bin/activate
```
### 2. install requirements
```
$ sudo apt install libmagickwand-dev
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
### 3. unzip & preprocess data
프로젝트 root dir에서 스크립트 입력
```
# 먼저, (데이터 압축파일) open.zip이 data경로 밑에 있어야 합니다!
$ bash shell_scripts/preprocess.sh
```

## Reproduction of results
## train models
프로젝트 root dir에서 스크립트 입력
```
$ bash shell_scripts/run_train_ensemble.sh
```
프로젝트 root dir에 모델 경로 생

## kfold inference & make result
프로젝트 root dir에서 스크립트 입력
```
$ bash shell_scripts/generate_result.sh
```
프로젝트 root dir에 result.csv가 생성된 것으로 제출.

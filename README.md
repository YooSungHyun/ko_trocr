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
& python3.8 -m venv .venv
& source .venv/bin/activate
```
### 2. install requirements
```
& pip install --upgrade pip
& pip install -r requirements.txt
```
### 3. unzip & preprocess data
```
& bash shell_scripts/preprocess.sh
```

## Reproduction of results
## train models
```
& bash shell_scripts/run_train_ensemble.sh
```

## kfold inference & make result
```
& bash sheell_scripts/generate_result.sh
```
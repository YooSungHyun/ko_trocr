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
├── arguments
│   ├── DatasetsArguments.py
│   ├── __init__.py
│   ├── ModelArguments.py
│   └── MyTrainingArguments.py
├── shell_scripts
│   ├── inference.sh
│   ├── install_base.sh
│   ├── pid_kill.sh
│   ├── preprocess.sh
│   └── run_train.sh
├── train.py
├── utils.py
├── preprocess.py
├── inference.py
├── requirements.txt
├── literal.py
├── LICENSE
└── README.md
```
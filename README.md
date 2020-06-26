# DefinitionExtraction
This repository focuses on Sequence Labelling and sentence classification task for definition extraction from text. Current work is inspired from: https://github.com/mattboggess/cs224n_glossary_extraction.   
Data is retrieved from multiple sources and is processsed by following [this](https://github.com/mattboggess/cs224n_glossary_extraction) repository with some minor changes.

## Files

```bash
.
├── data
│   ├── classification # sentence classification data
│   │   ├── all 
│   │   ├── w00 
│   │   └── wcl 
│   ├── tagging # sentence tagging data
│   │   └──openstax # data from openstax textbooks
├── src
│   ├── anke
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── sentence_classifier.py
│   │   └── train.py
│   ├── anke
│   │   ├── __init__.py
│   │   ├── cnn_blstm_crf.py
│   │   ├── data_loader.py
│   │   └── train.py
│   └── __init__.py
├── utils
│   ├── params_anke_all.json # classification model hyperparameters to be trained on all data
│   ├── params_anke_w00.json # classification model hyperparameters to be trained on w00 data
│   ├── params_anke_wcl.json # classification model hyperparameters to be trained on wcl data
│   └── params_hovy_openstax.json # tagging model hyperparameters to be trained on openstax data. cnn blstm csf
├── __init__.py
├── .gitignore
├── environment.yml
├── main.py
├── README.md
└── requirements.txt
```

### Installing
Create virtual environment (You can use python virtual environment using requirements.txt, I prefer conda)
```
conda env create -f environment.yml
```
or
```
conda create --name <env_name> python=3.5 --file requirements.txt
```
NOTE: Replace <env_name> with your choice of environment name or change name in environment.yml file if using .yml file.

### Activate
Activate the environment
```
conda activate <env_name>
```
use def as <env_name> if directly installed using .yml file.

### Run the file as
```
python main.py -M mode[train/test] -m model_save_path
```
for more info
```
python main.py -h
```
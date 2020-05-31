# DefinitionExtraction
This repository focuses on Sequence Labelling and sentence classification task for definition extraction from text. Current work is inspired from: https://github.com/mattboggess/cs224n_glossary_extraction. 

## Files

```bash
.
├── data - data retrieved from openstax textbooks using [this](https://github.com/mattboggess/cs224n_glossary_extraction) repository
│   ├── classification  - sentence classification data
│   │   ├── all 
│   │   ├── w00 
│   │   └── wcl 
│   ├── tagging  - sentence tagging data
│   │   └──openstax - data from openstax textbooks
├── src
│   ├── models
│   │   ├── __init__.py
│   │   └── sentence_classifier.py
│   ├── __init__.py
│   ├── data_loader.py
│   └── train.py
├── utils
│   └── params_classification.json - classification model hyperparameters.
├── __init__.py
├── .gitignore
├── main.py
└── README.md
```

### Installing
Create virtual environment (You can use python virtual environment using requirements.txt), I prefer conda)
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
python main.py -s model_save_path
```
for more info
```
python main.py -h
```
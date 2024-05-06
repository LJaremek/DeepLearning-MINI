# DeepLearning-MINI
Project 1 from the lecture "Deep Learning" at the MINI faculty at WUT.

## 1. Project description
Train student model based on trained teacher model.

## 2. Project structure
```
.
├── data                    <-- dataset for training and evaluate models
│   ├── train               <-- train dataset with classes folders
│   │   ├── automobile
│   │   ├── bird
│   │   ├── cat
│   │   ├── deer
│   │   ├── dog
│   │   ├── frog
│   │   ├── ship
│   │   └── truck
│   │
│   └── train               <-- test dataset with classes folders
│       ├── automobile
│       ├── bird
│       ├── cat
│       ├── deer
│       ├── dog
│       ├── frog
│       ├── ship
│       └── truck
│
├── models_status           <-- models state dicts
│
├── calc_data_std_mean.py   <-- calc std and mean of dataset script
├── data_mean_std.json      <-- calculated std and mean of dataset
├── evaluate_models.py      <-- evaluate models for checking their accuracy
├── models.py               <-- models instantions
├── README.md
├── tools.py                <-- universal auxiliary functions
├── train_student_model.py  <-- train student model script
└── train_teacher_model.py  <-- train teacher model script
```


## 3. Data source
Download the data and put the folders 'train' and 'test' to the 'data' folder in project repository.
https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y


## 4. How to run code

### 4.1. Install requirements:
```bash
pip install -r requirements.txt
```

### 4.2. Evalueate models
For evaluate models run `evaluate_models.py`

### 4.3. Train your own models
Run `train_teacher_model.py` and then run `train_student_model.py`

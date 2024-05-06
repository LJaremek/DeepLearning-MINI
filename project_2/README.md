# DeepLearning-MINI
Project 2 from the lecture "Deep Learning" at the MINI faculty at WUT.

## 1. Project description
Using transformer and other models for voice commands detection.

## 2. Project structure
```
.
├── data                        <-- dataset for training and evaluate models
│   ├── train                   <-- train dataset with classes folders
│   │   ├── up
│   │   ├── down
│   │   ├── left
│   │   └── right
│   │
│   └── background_noise        <-- dataset with noise files
│
├── models                      <-- models pth files
│
├── classes.py                  <-- torch models classes
├── models_experiments.ipynb    <-- calculated std and mean of dataset
├── robot.py                    <-- Robot class for simulate real robot 
├── robot_example_usage.py      <-- exaple using Robot class in real time
├── test-real_time.py           <-- testing voice commands detecting model
├── voice_control_robot.py      <-- control robot by voice commands
└── README.md
```


## 3. Data source
Download the data and put the 'train' folder
https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge


## 4. How to run code

### 4.1. Install requirements:
```bash
pip install -r requirements.txt
```

### 4.2. Train your own models
Run `model_experiments.ipynb`

### 4.3. Control robot by voice commands
Run `voice_control_robot.py` and say `up`, `down`, `left` or `right`

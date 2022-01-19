# Neural Clinical Event Sequence Prediction through Personalized Online Adaptive Learning

This repository is the official implementation of Neural Clinical Event Sequence Prediction through Personalized Online Adaptive Learning. The paper is published at 19th International Conference on Artificial Intelligence in Medicine (AIME) 2021. Link: https://arxiv.org/pdf/2104.01787.pdf


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data Preparation
This paper experiments with MIMIC-3 Clinical Database, provided by physionet.org. 
As they prohibit re-distribution of the MIMIC-3 database, we cannot provide preprocessed dataset we use for our experiment.

Instead, we prepare the data extraction, event-time-series generation, and featurization code in this repository.

**Prerequsite**
(1) Obtain MIMIC-3 access at https://mimic.physionet.org/gettingstarted/access/
(2) Obtain the database (csv files) and install them at your local database (e.g., MySQL)
(3) Setting database access info (account, password) at `prep_data/DBConnectInfo.py`.

**Training/Test Data Generation**
(1) Set proper `data path` according to your local computer setting in following files:
* `prep_data/prep_mimic_extract_ts_adhoc.py`
* `prep_data/scripts/exec_prep_mimic_gen_seq_lab_range_split10_mimicid.sh`
* `prep_data/scripts/exec_prep_mimic_remap_itemid_split10.bash`

(2) Run scripts according to `prep_data/scripts/bootstrap.bash`. I recommend to run each step in the file one by one.


## Training and Evaluation

We recommend to train `GRU-POP` model first. Then, provide the path of the trained model of the GRU-POP at `--load-model-from` argument.

When one experiment is run, it first do training and then it automatically run evaluation.

In order to run evaluation only, provide this argument: `--eval-only --load-model-from [pretrained model]`.


To train and get the evaluation the models in the paper, run following commands in `scripts` folder:

### GRU-POP
```
bash run_sci.bash 24 GRU 1 "--fast-folds 1--hyper-weight-decay 1e-04 --hyper-weight-decay 1e-05 --hyper-weight-decay 1e-06 --hyper-weight-decay 1e-07 --bptt 0 --eval-on-cpu --hidden-dim 512 --multiproc 1 --learning-rate 0.005";
```   

### RETAIN
```
bash run_sci.bash 24 RETAIN 1 "--fast-folds 1 --hyper-weight-decay 1e-04 --hyper-weight-decay 1e-05 --hyper-weight-decay 1e-06 --hyper-weight-decay 1e-07 --bptt 0 --eval-on-cpu --hidden-dim 512 --multiproc 1 --learning-rate 0.005";
```  

### CNN
```
bash run_sci.bash 24 CNN 1 "--fast-folds 1--hyper-weight-decay 1e-04 --hyper-weight-decay 1e-05 --hyper-weight-decay 1e-06 --hyper-weight-decay 1e-07 --bptt 0 --eval-on-cpu --hidden-dim 512 --multiproc 1 --learning-rate 0.005";
```  

### GRU-IN 
```
bash run_sci.bash 24 GRU 1 "--skip-hypertuning --weight-decay 1e-05 --hidden-dim 512 --multiproc 2 --bptt 0 --eval-on-cpu --eval-only --load-model-from pretrained_model/GRU-POP.model --adapt-loss bce --adapt-lr 0.0005 --adapt-bandwidth 3 --adapt-lstm";
``` 

### GRU-IN-SW 
```
bash run_sci.bash 24 GRU 1 "--skip-hypertuning --weight-decay 1e-05 --hidden-dim 512 --multiproc 2 --bptt 0 --eval-on-cpu --eval-only --load-model-from pretrained_model/GRU-POP.model --adapt-loss bce --adapt-lr 0.0005 --adapt-bandwidth 3 --adapt-lstm --adapt-switch";
``` 

### GRU-IN-AO-SW 
```
bash run_sci.bash 24 GRU 1 "--skip-hypertuning --weight-decay 1e-05 --hidden-dim 512 --multiproc 2 --bptt 0 --eval-on-cpu --eval-only --load-model-from pretrained_model/GRU-POP.model --adapt-loss bce --adapt-lr 0.0005 --adapt-bandwidth 3 --adapt-fc-only --adapt-switch";
``` 



## Trained (Output) Models

The trained models that used to generate experiment reports are located under `pretrained_models` folder.

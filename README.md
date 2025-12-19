# IR-STDGCN

**[Official Implementation of]** Incremental Regularization-Enhanced Spatial-Temporal Decoupling Graph Convolutional Network for Class-Incremental Skeleton-based Gesture Recognition

## Dependencies

### Requirements
- Python >= 3.8
- PyTorch >= 1.8
- CUDA >= 11.0

### Installation
```bash
# Clone the repository
git clone https://github.com/QienJane/IR-STDGCN.git
cd IR-STDGCN

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### SHREC'17 Track
The SHREC'17 Track dataset contains 14 hand gesture classes from 28 subjects, with 22 key joints per gesture.

1. Download the dataset from the [website](http://www-rech.telecom-lille.fr/shrec2017-hand/).
2. Extract files to `./data/shrec/`
3. Run preprocessing:

```bash
cd data/shrec
python gen_traindataset.py
python gen_testdataset.py
```

### EgoGesture3D
The EgoGesture3D dataset contains hand gestures from 50 subjects collected from a first-person perspective.

1. Download the dataset from the [website](https://drive.google.com/file/d/1pHE0Q9MtVS5BLaV2CBN1rLP_Ed7nvfac/view).
2. Extract files to `./data/egogesture3d/`
3. Run preprocessing:

```bash
cd data/egogesture3d
python egogesture3d.py
```

### NTU RGB+D 60
The NTU RGB+D 60 dataset includes 60 action classes from 40 subjects, with 56,880 skeleton sequences (25 joints).

1. Download the dataset from the [website](https://rose1.ntu.edu.sg/dataset/actionRecognition/).
2. Extract files to `./data/ntu`
3. Run preprocessing:

```bash
cd data/ntu
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

### NTU RGB+D 120
The NTU RGB+D 120 dataset extends NTU-60 to 120 classes with 114,480 sequences from 106 subjects.

1. Download the dataset from the [website](https://rose1.ntu.edu.sg/dataset/actionRecognition/).
2. Follow the same preprocessing steps as NTU-60

### NW-UCLA
The Northwestern-UCLA dataset contains 10 action classes with 1,494 skeleton samples (20 joints).

1. Download the dataset from the [website](https://wangjiangb.github.io/my_data.html).
2. Extract files to `./data/nw-ucla/`

## Training

### Start
Run experiments with the provided script:
```bash
bash scripts/run_experiments.sh
```

### Script Configuration
Before running, modify the paths in `scripts/run_experiments.sh`:
- `project_dir`: Path to the project root directory
- `scripts_dir`: Path to the scripts directory

And in `scripts/run_trial.sh`:
- `root_dir`: Path to the dataset directory
- `cfg_file`: Path to config file
- `log_dir`: Output directory

## Testing

### Evaluation
```bash
cd drivers
python main_driver.py \
    --train -1 \
    --dataset shrec \
    --cfg_file ../configs/params/dataset/IR.yaml \
    --test_only
```

## Project Structure
```
IR-STDGCN/
├── data/                      # Dataset directory
│   ├── shrec/                 # SHREC'17 dataset
│   ├── egogesture3d/          # EgoGesture3D dataset
│   ├── ntu/                   # NTU RGB+D dataset
│   └── nw-ucla/               # NW-UCLA dataset
├── drivers/                   # Main drivers
├── feeders/                   # Data loaders
├── graph/                     # Graph structure definitions
├── learners/                  # Incremental learning methods
├── losses/                    # Loss functions
├── model_defs/                # Model definitions
├── optimizers/                # Optimizers
├── scripts/                   # Running scripts
└── utils/                     # Utility functions
```

## Acknowledgements
This project is based on [BOAT-MI](https://github.com/humansensinglab/dfcil-hgr) and [TD-GCN](https://github.com/liujf69/TD-GCN-Gesture). We thank the authors for their excellent work.

## Contact
For any questions, please contact zhanqq@mail2.sysu.edu.cn
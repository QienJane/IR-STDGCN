import sys
import os, os.path as osp

import argparse
import yaml
import importlib
from easydict import EasyDict as edict

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
torch.use_deterministic_algorithms(True)

if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))

SUB_DIR_LEVEL = 1 # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)))

import utils

sys.path.append('/.../IR-STDGCN')

try:
    import graph
except ModuleNotFoundError as e:
    print(f"Error: {e}")


parser = argparse.ArgumentParser(description='Gesture Recognition.')
parser.add_argument('--train', type=int, default=1, required=False, help='train (1) or testval (0) or test (-1).')
parser.add_argument('--dataset', type=str, default='SHREC2017', required=False, help='name of the dataset.')
parser.add_argument('--split_type', type=str, default='agnostic', required=False, help='type of data split (if applicable).')
parser.add_argument('--cfg_file', type=str, default='/.../IR-STDGCN/configs/params/shrec/IR.yaml', required=False, help='config file to load experimental parameters.')
parser.add_argument('--root_dir', type=str, default='/.../IR-STDGCN/SHREC2017', required=False, help='root directory containing the dataset.')
parser.add_argument('--log_dir', type=str, default='/.../IR-STDGCN/output/shrec', required=False, help='directory for logging.')
parser.add_argument('--save_last_only', action='store_true', help='whether to save the last epoch only.')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='epoch frequency to save checkpoints.')
parser.add_argument('--save_conf_mat', action='store_true', default=1, help='whether to save the confusion matrix.')
parser.add_argument('--gpu', type=int, default=1, help='GPU ID.')
parser.add_argument('--trial_id', type=int, default=5, help='trial_ID')


def main() :
    args = parser.parse_args()
    args.dist_url = 'tcp://127.0.0.1:' + utils.get_free_port()
    print('args.gpu:',args.gpu)
    print('--------------------Begin the experiment:')
    utils.print_argparser_args(args)
    utils.set_seed()
    n_gpus = torch.cuda.device_count()

    if n_gpus <= 0:
        raise AssertionError("A GPU is required for execution.")
    main_worker(args.gpu, 1, args)

def main_worker(gpu, n_gpus, args) :
    with open(args.cfg_file, 'rb') as f :
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    cfg_data = getattr(importlib.import_module('.' + args.dataset, package='configs.datasets'),
                'Config_Data')(args.root_dir)

    is_distributed = False
    is_train = (args.train==1)

    # Learners dict
    learners_dict = { 'ir': 'IncrementalRegularization'}

    # Execute trial
    root_log_dir = args.log_dir
    trial_id = args.trial_id
    with open(args.cfg_file, 'rb') as f :
            cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    print(f'--------------------Executing trial_{trial_id+1}--------------------')

    trial_log_dir = osp.join(root_log_dir, f'trial_{trial_id+1}')
    if not osp.exists(trial_log_dir) :
        os.makedirs(trial_log_dir)
    args.log_dir = trial_log_dir

    learner = getattr(importlib.import_module('.' + cfg.increm.learner.type, package='learners'),
                learners_dict[cfg.increm.learner.type])(cfg, cfg_data, args, is_train, is_distributed, n_gpus)

    if is_train :
        learner.train(n_trial=trial_id)
    else:
        # Evaluate
        learner.evaluate(n_trial=trial_id)

if __name__ == '__main__' :
    main()

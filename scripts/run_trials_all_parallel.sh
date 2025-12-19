#!/bin/bash

# Run multiple trials in parallel, suitable for running multiple trial_ids simultaneously
# Run experiment
project_dir=/.../IR-STDGCN
scripts_dir=/.../scripts
cd ${scripts_dir}

# General config
split_type="agnostic"
CUDA_VISIBLE_DEVICES=0
gpu=0

datasets=("shrec")

increment=("IR")
n_trials=3
n_tasks=7

#Run trials
trial_id=0
./run_trial.sh $trial_id $project_dir $split_type $gpu "${datasets[*]}" "${increment[*]}" &

trial_id=1
./run_trial.sh $trial_id $project_dir $split_type $gpu "${datasets[*]}" "${increment[*]}" &

trial_id=2
./run_trial.sh $trial_id $project_dir $split_type $gpu "${datasets[*]}" "${increment[*]}" &

#!/bin/bash
# Get and store start time
start_time=$(date +%s)
start_readable=$(date +"%Y-%m-%d %H:%M:%S")
start_msg="Program start time: $start_readable"

# Print program start time
printf "%s\n" "$start_msg"

# Run experiment
project_dir=/.../IR-STDGCN
scripts_dir=/.../scripts
cd ${scripts_dir}

# General config
split_type="agnostic"

datasets=("shrec")

increment=("IR")
trial_ids=(0)
n_trials=${#trial_ids[@]}
n_tasks=7

#Run trials

CUDA_VISIBLE_DEVICES=0
gpu=0
for trial_id in ${trial_ids[*]}; do
    ./run_trial.sh $trial_id $project_dir $split_type $gpu "${datasets[*]}" "${increment[*]}"
done


end_time=$(date +%s)
end_readable=$(date +"%Y-%m-%d %H:%M:%S")
end_msg="Program end time: $end_readable"
total_time=$((end_time - start_time))

hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))
duration_msg="Total duration: ${hours}h ${minutes}m ${seconds}s"

printf "%s\n" "$start_msg" "$end_msg" "$duration_msg"

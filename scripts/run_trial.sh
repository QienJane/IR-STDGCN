#!/bin/bash

# Run trial
trial_id=$1
project_dir=$2
split_type=$3
gpu=$4
datasets=$5
increment=$6

cd ${project_dir}/drivers

run_driver() {

    python main_driver.py \
    --train ${train} \
    --dataset ${dataset} \
    --split_type ${split_type} \
    --cfg_file ${cfg_file} \
    --root_dir ${root_dir} \
    --log_dir ${log_dir} \
    --gpu ${gpu} \
    --save_last_only \
    --trial_id ${trial_id}
}

for dataset_name in ${datasets[*]}; do
    if [ $dataset_name = "shrec" ]
    then
        dataset="shrec"
        root_dir="/.../SHREC2017"
    elif [ $dataset_name = "egogesture3d" ]
    then
        dataset="egogesture3d"
        root_dir="/.../ego_gesture_v4"
    elif [ $dataset_name = "ntu" ]
    then
        dataset="ntu"
        root_dir="/.../ntu"
    elif [ $dataset_name = "nw-ucla" ]
    then
        dataset="nw-ucla"
        root_dir="/.../nw-ucla"
    fi

    for increment_name in ${increment[*]}; do
        ############################ Run increment ############################
        #Train
        train=1
        cfg_file=/.../IR-STDGCN/configs/params/$dataset/$increment_name.yaml
        log_dir=/.../IR-STDGCN/output/$dataset/$increment_name
        trial_id=$trial_id
        run_driver

        #Test
        train=-1
        run_driver
    done
done
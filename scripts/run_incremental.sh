#!/bin/bash
DATASET=${1:-"shrec"}
GPU_ID=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU_ID

CONFIG_PATH="configs/params/${DATASET}/IR.yaml"

# Determine number of tasks based on dataset
case $DATASET in
    "shrec")
        MAX_TASK=6
        ;;
    "egogesture3d")
        MAX_TASK=6
        ;;
    "ntu60")
        MAX_TASK=6
        ;;
    "ntu120")
        MAX_TASK=12
        ;;
    "ucla")
        MAX_TASK=4
        ;;
    *)
        MAX_TASK=6
        ;;
esac

echo "Training $((MAX_TASK + 1)) tasks (0 to $MAX_TASK)..."

# Train all tasks sequentially
for TASK_ID in $(seq 0 $MAX_TASK); do

    python -m drivers.main_driver \
        --config $CONFIG_PATH \
        --task_id $TASK_ID

    if [ $? -ne 0 ]; then
        echo "Error in Task $TASK_ID. Stopping."
        exit 1
    fi
done
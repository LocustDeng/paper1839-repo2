#!/bin/bash

dirs=("******/dataset/LLFF" "******/dataset/mip360" "******/dataset/my-flower")

for base_dir in "${dirs[@]}"; do
    for sub_dir in "$base_dir"/*; do
        if [ -d "$sub_dir" ]; then
            INPUT_DIR="$sub_dir"
            OUTPUT_DIR="$sub_dir/******"
            NOHUP_FILE_1="******/experiment/******/train_log_n"
            NOHUP_FILE_2="******/experiment/******/render_log_n"
            NOHUP_FILE_3="******/experiment/******/eval_log_n"
            echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>"
            echo "正在训练: $INPUT_DIR"
            python train.py -s "$INPUT_DIR" -m "$OUTPUT_DIR" --eval > "$NOHUP_FILE_1" 2>&1
            echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>"
        else
            echo "跳过 $sub_dir，因为它不是一个目录"
        fi
    done
done
#!/bin/bash

dirs=("******/dataset/LLFF" "******/dataset/mip360" "******/dataset/my-flower")
NOHUP_FILE_1="******/experiment/******/train_log_n"
NOHUP_FILE_2="******/experiment/******/render_log_n"
NOHUP_FILE_3="******/experiment/******/eval_log_n"

for base_dir in "${dirs[@]}"; do
    for sub_dir in "$base_dir"/*; do
        if [ -d "$sub_dir" ]; then
            INPUT_DIR="$sub_dir"
            OUTPUT_DIR="$sub_dir/******"
            echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>"
            echo "正在渲染：$INPUT_DIR"
            # python render.py -m "$OUTPUT_DIR" -s "$INPUT_DIR" > "$NOHUP_FILE_2" 2>&1
            python render.py -m "$OUTPUT_DIR" -s "$INPUT_DIR" >> "$NOHUP_FILE_2" 2>&1
            wait
            echo "正在评估：$INPUT_DIR"
            # python metrics.py -m "$OUTPUT_DIR" > "$NOHUP_FILE_3" 2>&1
            python metrics.py -m "$OUTPUT_DIR" >> "$NOHUP_FILE_3" 2>&1
            echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>"
        else
            echo "跳过 $sub_dir，因为它不是一个目录"
        fi
    done
done
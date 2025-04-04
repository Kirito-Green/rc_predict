#! /bin/bash


lrs=(1e-4)
model_names=("gcn" "graph_sage" "gat")

for model_name in ${model_names[@]}; do
    for lr in ${lrs[@]}; do
        python -X faulthandler demo_gnn.py \
        --dir_prj "/home/prj/rc_predict/" \
        --seed 42 \
        --pattern_nums -1 \
        --thresh 0 3000 \
        --num_process 50 \
        --model_name $model_name \
        --lr $lr \
        --batch_size 32 \
        --epochs 100 \
        --load_params False \
        --set_memory_growth True \
        --set_multi_gpu_num 1
    done
done

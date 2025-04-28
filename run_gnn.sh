#! /bin/bash


lrs=(1e-3)
model_names=("gcn" "gat")
batch_sizes=(32)
ndms=(25)

for model_name in ${model_names[@]}; do
	for lr in ${lrs[@]}; do
		for batch_size in ${batch_sizes[@]}; do
			for ndm in ${ndms[@]}; do
				python -X faulthandler demo_gnn.py \
				--dir_prj "/home/prj/rc_predict/" \
				--seed 42 \
				--pattern_nums -1 \
				--thresh 0 3000 \
				--num_process 32 \
				--ndm $ndm \
				--model_name $model_name \
				--lr $lr \
				--batch_size $batch_size \
				--epochs 100 \
				--load_params False \
				--set_memory_growth True \
				--set_multi_gpu_num 1
			done
		done
	done
done

#! /bin/bash


function extract_cap(){
	model_names=$1
	lrs=$2
	batch_sizes=$3
	ks=$4

	for model_name in ${model_names[@]}; do
		for lr in ${lrs[@]}; do
			for batch_size in ${batch_sizes[@]}; do
				for k in ${ks[@]}; do
					python -X faulthandler demo_gnn_ref.py \
					--dir_prj "/home/prj/rc_predict/" \
					--seed 42 \
					--pattern_nums 3 4 26 \
					--nodes_range 50 1500 \
					--num_process 32 \
					--k $k \
					--model_name $model_name \
					--lr $lr \
					--batch_size $batch_size \
					--epochs 100 \
					--load_params False \
					--set_memory_growth True \
					--set_multi_gpu_num 3 \
					-nt
				done
			done
		done
	done
}


# main experiment
# model_names=("gcn", 'gat')
# lrs=(3e-4)
# batch_sizes=(32)
# ks=(15)
# extract_cap $model_names $lrs $batch_sizes $ks

# else experiment
model_names=("gat")
lrs=(1e-3)
batch_sizes=(16)
ks=(20)
extract_cap $model_names $lrs $batch_sizes $ks

#! /bin/bash

lrs=(1e-3 5e-4 2e-4 1e-4)
model_names=("gcn" "graph_sage" "gat")

for model_name in ${model_names[@]}; do
    echo ${model_name}
done


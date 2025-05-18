#! /bin/bash


python demo_base.py \
--dir_prj "/public/home/ypx/workspace/prj/rc_predict/" \
--seed 42 \
--pattern_nums -1 \
--nodes_range 0 3000 \
--num_process 32 \
--n_components 100 \
--disable_ddr

#!/bin/bash
module load compiler/cuda/10.0/compilervars
module load compiler/gcc/6.5.0/compilervars
python train.py --s_ds ffhq --t_ds babies --exp babies --npy_file ../data/babies_training.npy --n_shot 10 --iter 251 --img_freq 250 --adv_bs 8 --sty_bs 10 --start 0 --stop 501
python compute_fid.py --t_ds babies --start 0 --stop 501

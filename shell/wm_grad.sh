#!/usr/bin/env bash
eps=0.2
eps_iter=5
pert_type='2'
nb_iter=1000
font_name=('Courier' 'Georgia' 'Helvetica' 'times' 'Arial')

#case='replace-full-word'
#eps_iter=0.5
#nb_iter=2000
case='easy'
CUDA_VISIBLE_DEVICES=3 python wm_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
CUDA_VISIBLE_DEVICES=5 python wm_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &

#CUDA_VISIBLE_DEVICES=5 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter 230 & # 0.9
#eps_iter=1
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter 204 & # 0.8
#eps_iter=2
#CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter 179 & # 0.7
#eps_iter=10
#CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter 242 & # 0.95
#CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &
#case='easy'
#CUDA_VISIBLE_DEVICES=5 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &
#case='hard'
#case='delete'
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &
#case='replace-full-word'
#case='insert'
#CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &



#case='easy'
#CUDA_VISIBLE_DEVICES=1 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &
#case='random'
#CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &
#case='hard'
#CUDA_VISIBLE_DEVICES=5 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &
#case='delete'
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &
#case='insert'
#CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &
#case='replace-full-word'
#CUDA_VISIBLE_DEVICES=3 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &

#case='replace'
#CUDA_VISIBLE_DEVICES=0 python wm_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=1 python wm_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=2 python wm_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=3 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &

#case='delete'
#CUDA_VISIBLE_DEVICES=5 python wm_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &

#case='insert'
#CUDA_VISIBLE_DEVICES=3 python wm_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=1 python wm_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &

#case='replace-full-word'
#CUDA_VISIBLE_DEVICES=2 python wm_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=1 python wm_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &

#case='word'
#nb_iter=1000
#CUDA_VISIBLE_DEVICES=0 python wm_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=1 python wm_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=3 python wm_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=5 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &

#case='sentence'
#CUDA_VISIBLE_DEVICES=0 python wm_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=1 python wm_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=2 python wm_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=3 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &

#case='protect'
#nb_iter=2000
#CUDA_VISIBLE_DEVICES=3 python wm_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=5 python wm_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &



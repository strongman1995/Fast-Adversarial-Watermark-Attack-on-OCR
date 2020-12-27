#!/usr/bin/env bash
eps=0.2
eps_iter=5
pert_type='2'
nb_iter=2000
font_name=('Courierbd' 'Georgiabd' 'Helveticabd' 'timesbd' 'Arialbd')

#case='easy'
#case='random'
#case='hard'

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
#CUDA_VISIBLE_DEVICES=2 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=3 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &
#
#case='insert'
#CUDA_VISIBLE_DEVICES=4 python wm_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=5 python wm_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=2 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &

#case='replace-full-word'
#CUDA_VISIBLE_DEVICES=3 python wm_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=7 python wm_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=5 python wm_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=6 python wm_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &



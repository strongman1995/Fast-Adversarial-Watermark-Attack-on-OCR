#!/usr/bin/env bash
eps=0.2
eps_iter=5
pert_type='2'
nb_iter=1000
font_name=('Courier' 'Georgia' 'Helvetica' 'times' 'Arial')

case='easy'
#CUDA_VISIBLE_DEVICES=3 python long_attack.py ${font_name[0]} $case $pert_type $eps $eps_iter $nb_iter &
CUDA_VISIBLE_DEVICES=4 python long_attack.py ${font_name[1]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=5 python long_attack.py ${font_name[2]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=6 python long_attack.py ${font_name[3]} $case $pert_type $eps $eps_iter $nb_iter &
#CUDA_VISIBLE_DEVICES=7 python long_attack.py ${font_name[4]} $case $pert_type $eps $eps_iter $nb_iter &

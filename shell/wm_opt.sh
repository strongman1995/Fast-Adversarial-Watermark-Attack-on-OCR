#!/usr/bin/env bash
font_name=('Courier' 'Georgia' 'Helvetica' 'times' 'Arial')

nb_iter=1000
#case='easy'
#const=5
#CUDA_VISIBLE_DEVICES=3 python wm_opt_attack.py ${font_name[4]} $case $nb_iter $const&
#const=0.5
#CUDA_VISIBLE_DEVICES=4 python wm_opt_attack.py ${font_name[4]} $case $nb_iter $const&
#const=25
#CUDA_VISIBLE_DEVICES=5 python wm_opt_attack.py ${font_name[4]} $case $nb_iter $const&
#const=10
#CUDA_VISIBLE_DEVICES=6 python wm_opt_attack.py ${font_name[4]} $case $nb_iter $const&
#const=1
#CUDA_VISIBLE_DEVICES=7 python wm_opt_attack.py ${font_name[4]} $case $nb_iter $const&

#case='random'
#CUDA_VISIBLE_DEVICES=4 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&
#case='hard'
#CUDA_VISIBLE_DEVICES=4 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&
#case='delete'
#CUDA_VISIBLE_DEVICES=5 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&
#case='insert'
#CUDA_VISIBLE_DEVICES=6 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&
#case='replace-full-word'
#CUDA_VISIBLE_DEVICES=6 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&


#nb_iter=1000
#case='easy'
#CUDA_VISIBLE_DEVICES=0 python wm_opt_attack.py ${font_name[0]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=1 python wm_opt_attack.py ${font_name[1]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=2 python wm_opt_attack.py ${font_name[2]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=3 python wm_opt_attack.py ${font_name[3]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=4 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&

#case='random'
#CUDA_VISIBLE_DEVICES=5 python wm_opt_attack.py ${font_name[0]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=6 python wm_opt_attack.py ${font_name[1]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=7 python wm_opt_attack.py ${font_name[2]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=0 python wm_opt_attack.py ${font_name[3]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=1 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&
#
#case='hard'
#CUDA_VISIBLE_DEVICES=2 python wm_opt_attack.py ${font_name[0]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=3 python wm_opt_attack.py ${font_name[1]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=4 python wm_opt_attack.py ${font_name[2]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=5 python wm_opt_attack.py ${font_name[3]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=6 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&

#nb_iter=2000
#case='insert'
#CUDA_VISIBLE_DEVICES=7 python wm_opt_attack.py ${font_name[0]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=0 python wm_opt_attack.py ${font_name[1]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=1 python wm_opt_attack.py ${font_name[2]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=2 python wm_opt_attack.py ${font_name[3]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=3 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&

#case='delete'
#CUDA_VISIBLE_DEVICES=4 python wm_opt_attack.py ${font_name[0]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=5 python wm_opt_attack.py ${font_name[1]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=6 python wm_opt_attack.py ${font_name[2]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=7 python wm_opt_attack.py ${font_name[3]} $case $nb_iter&
#CUDA_VISIBLE_DEVICES=0 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&

case='replace-full-word'
nb_iter=2000
CUDA_VISIBLE_DEVICES=3 python wm_opt_attack.py ${font_name[0]} $case $nb_iter&
CUDA_VISIBLE_DEVICES=4 python wm_opt_attack.py ${font_name[1]} $case $nb_iter&
CUDA_VISIBLE_DEVICES=5 python wm_opt_attack.py ${font_name[2]} $case $nb_iter&
CUDA_VISIBLE_DEVICES=6 python wm_opt_attack.py ${font_name[3]} $case $nb_iter&
CUDA_VISIBLE_DEVICES=7 python wm_opt_attack.py ${font_name[4]} $case $nb_iter&


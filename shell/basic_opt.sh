#!/usr/bin/env bash
font_name=('Courier' 'Georgia' 'Helvetica' 'times' 'Arial')

case='easy'
const=0.5
CUDA_VISIBLE_DEVICES=3 python basic_opt_attack.py ${font_name[4]} $case $const&
const=5
CUDA_VISIBLE_DEVICES=4 python basic_opt_attack.py ${font_name[4]} $case $const&
#const=25
#CUDA_VISIBLE_DEVICES=5 python basic_opt_attack.py ${font_name[4]} $case $const&
#const=10
#CUDA_VISIBLE_DEVICES=6 python basic_opt_attack.py ${font_name[4]} $case $const&
#const=1
#CUDA_VISIBLE_DEVICES=7 python basic_opt_attack.py ${font_name[4]} $case $const&


#case='easy'
#CUDA_VISIBLE_DEVICES=5 python basic_opt_attack.py ${font_name[4]} $case &
#case='random'
#CUDA_VISIBLE_DEVICES=4 python basic_opt_attack.py ${font_name[4]} $case &
#case='hard'
#CUDA_VISIBLE_DEVICES=6 python basic_opt_attack.py ${font_name[4]} $case &
#case='delete'
#CUDA_VISIBLE_DEVICES=6 python basic_opt_attack.py ${font_name[4]} $case &
#case='insert'
#CUDA_VISIBLE_DEVICES=7 python basic_opt_attack.py ${font_name[4]} $case &
#case='replace-full-word'
#CUDA_VISIBLE_DEVICES=7 python basic_opt_attack.py ${font_name[4]} $case &

#case='easy'
#CUDA_VISIBLE_DEVICES=3 python basic_opt_attack.py ${font_name[0]} $case &
#CUDA_VISIBLE_DEVICES=4 python basic_opt_attack.py ${font_name[1]} $case &
#CUDA_VISIBLE_DEVICES=5 python basic_opt_attack.py ${font_name[2]} $case &
#CUDA_VISIBLE_DEVICES=6 python basic_opt_attack.py ${font_name[3]} $case &
#CUDA_VISIBLE_DEVICES=7 python basic_opt_attack.py ${font_name[4]} $case &

#case='random'
#CUDA_VISIBLE_DEVICES=3 python basic_opt_attack.py ${font_name[0]} $case &
#CUDA_VISIBLE_DEVICES=4 python basic_opt_attack.py ${font_name[1]} $case &
#CUDA_VISIBLE_DEVICES=5 python basic_opt_attack.py ${font_name[2]} $case &
#CUDA_VISIBLE_DEVICES=6 python basic_opt_attack.py ${font_name[3]} $case &
#CUDA_VISIBLE_DEVICES=7 python basic_opt_attack.py ${font_name[4]} $case &

#case='hard'
#CUDA_VISIBLE_DEVICES=3 python basic_opt_attack.py ${font_name[0]} $case &
#CUDA_VISIBLE_DEVICES=4 python basic_opt_attack.py ${font_name[1]} $case &
#CUDA_VISIBLE_DEVICES=5 python basic_opt_attack.py ${font_name[2]} $case &
#CUDA_VISIBLE_DEVICES=6 python basic_opt_attack.py ${font_name[3]} $case &
#CUDA_VISIBLE_DEVICES=7 python basic_opt_attack.py ${font_name[4]} $case &

#case='insert'
#CUDA_VISIBLE_DEVICES=3 python basic_opt_attack.py ${font_name[0]} $case &
#CUDA_VISIBLE_DEVICES=4 python basic_opt_attack.py ${font_name[1]} $case &
#CUDA_VISIBLE_DEVICES=5 python basic_opt_attack.py ${font_name[2]} $case &
#CUDA_VISIBLE_DEVICES=6 python basic_opt_attack.py ${font_name[3]} $case &
#CUDA_VISIBLE_DEVICES=7 python basic_opt_attack.py ${font_name[4]} $case &

#case='delete'
#CUDA_VISIBLE_DEVICES=3 python basic_opt_attack.py ${font_name[0]} $case &
#CUDA_VISIBLE_DEVICES=4 python basic_opt_attack.py ${font_name[1]} $case &
#CUDA_VISIBLE_DEVICES=5 python basic_opt_attack.py ${font_name[2]} $case &
#CUDA_VISIBLE_DEVICES=6 python basic_opt_attack.py ${font_name[3]} $case &
#CUDA_VISIBLE_DEVICES=7 python basic_opt_attack.py ${font_name[4]} $case &

#case='replace-full-word'
#CUDA_VISIBLE_DEVICES=3 python basic_opt_attack.py ${font_name[0]} $case &
#CUDA_VISIBLE_DEVICES=4 python basic_opt_attack.py ${font_name[1]} $case &
#CUDA_VISIBLE_DEVICES=5 python basic_opt_attack.py ${font_name[2]} $case &
#CUDA_VISIBLE_DEVICES=6 python basic_opt_attack.py ${font_name[3]} $case &
#CUDA_VISIBLE_DEVICES=7 python basic_opt_attack.py ${font_name[4]} $case &


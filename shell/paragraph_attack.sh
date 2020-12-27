font_name=('Courier' 'Georgia' 'Helvetica' 'times' 'Arial')

CUDA_VISIBLE_DEVICES=3 python paragraph_attack.py ${font_name[0]} &
CUDA_VISIBLE_DEVICES=4 python paragraph_attack.py ${font_name[1]} &
CUDA_VISIBLE_DEVICES=5 python paragraph_attack.py ${font_name[2]} &
CUDA_VISIBLE_DEVICES=6 python paragraph_attack.py ${font_name[3]} &
CUDA_VISIBLE_DEVICES=7 python paragraph_attack.py ${font_name[4]} &